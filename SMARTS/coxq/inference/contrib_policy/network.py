import torch
from torch import nn
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path
import random
from functools import partial

from typing import Dict, List, Tuple, Type, Union, Optional, Any
from contrib_policy.make_dict import MakeObsDict

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
import gymnasium as gym
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import preprocess_obs


sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from contrib_policy.layers import *

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.neighbor_traj = SubGraph(c_in=5)
        self.ego_traj = nn.Sequential(nn.Linear(2, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256)
                                      )

    def forward(self, ego, neighbors, w):
        # ego_feature = torch.cat([ego, weight], -1)
        # ego_out = self.ego_traj(ego_feature)
        ego_out = self.ego_traj(ego)

        neighbors_out = self.neighbor_traj(neighbors)

        output = torch.cat([ego_out, neighbors_out], 1)
        return output

class AgentActionEncoder(nn.Module):
    def __init__(self):
        super(AgentActionEncoder, self).__init__()
        self.neighbor_traj = SubGraph(c_in=5)
        self.ego_traj = nn.Sequential(nn.Linear(4, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256)
                                      )

    def forward(self, ego, neighbors, action):
        # weight = w.unsqueeze(1).expand(ego.size(0), ego.size(1), 2)
        # print(ego.size(), action.size())
        ego_feature = torch.cat([ego, action.unsqueeze(1)], -1)
        ego_out = self.ego_traj(ego_feature)
        #ego_out = self.ego_traj(ego)

        neighbors_out = self.neighbor_traj(neighbors)

        output = torch.cat([ego_out, neighbors_out], 1)
        return output

class MapEncoder(nn.Module):
    def __init__(self):
        super(MapEncoder, self).__init__()
        self.splines = SubGraph(c_in=3)
        self.goal_net = nn.Sequential(nn.Linear(2, 256),
                                      nn.ReLU(), 
                                      nn.Linear(256, 256))

    def forward(self, map, goal):
        goal_encoded = self.goal_net(goal.unsqueeze(1))
        map_encoded = self.splines(map)
        output = torch.cat([goal_encoded, map_encoded], 1)
        return output

class Agent2Agent(nn.Module):
    def __init__(self,):
        super(Agent2Agent, self).__init__()
        self.interaction_1 = SelfTransformer()
        self.interaction_2 = SelfTransformer()

    def forward(self, inputs, mask=None):
        output = self.interaction_1(inputs, mask=mask)
        output = self.interaction_2(inputs+output, mask=mask)

        return output

class ActionFeatureNet(nn.Module):
    def __init__(self, 
                 action_dim: int,
                 ):
        super(ActionFeatureNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(action_dim, 128),
                                    nn.ReLU(), 
                                    nn.Linear(128, 64))

    def forward(self, x):
        return self.fc(x)
    
class ActionNet(nn.Module):
    def __init__(self, 
                 action_dim: int,
                 ):
        super(ActionNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(256+64, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, action_dim*2))

    def forward(self, x):
        return self.fc(x)

class FeatureNet_td3(nn.Module):
    def __init__(self, 
                 use_action,
                 action_dim,
                 for_value=False):#use_interaction):
        super(FeatureNet_td3, self).__init__()

        self.for_value = for_value 
        self.use_action = use_action

        # agent/map layer
        self.agent_net= AgentActionEncoder()
        self.map_net = MapEncoder()

        # self.bev_net =CNNExtractor()
        
        # attention layers
        self.agent_agent = Agent2Agent()
        #self.map2agent = CrossTransformer()
        self.fuse = SelfTransformer()

        self.preference =  nn.Sequential(nn.Linear(2, 128), 
                                         nn.ReLU(), 
                                         nn.Linear(128, 64))
        
        self.actionnet = ActionFeatureNet(action_dim)

    def forward(self, observations, actions):
        ego = observations['ego_state']
        neighbors = observations['neighbors_state']
        map = observations['ego_map']
        # bev = observations['bev']
        goal = observations['goal']
        w = observations['weight']

        #action_feature = self.actionnet(actions)
        # bev_feature = self.bev_net(bev)

        # mask generation
        agent_mask = torch.eq(torch.sum(torch.cat([neighbors[:,:1], 
                                                   neighbors], 1)[:,:,-1], -1), 0)

        agent_mask[:, 0] = False
        map_mask = torch.eq(torch.sum(torch.cat([map[:,:1], 
                                                 map], 1), (-2,-1)), 0)
        map_mask[:, 0] = False
	    # actionmask = torch.zeros_like(map_mask[:, :1])
        # print(map_mask.shape, agent_mask.shape)
        global_mask = torch.cat([agent_mask, map_mask], -1)

        # element-wise features
        agent_encoded = self.agent_net(ego, neighbors, actions)
        map_encoded = self.map_net(map, goal)

        agent_agent = self.agent_agent(agent_encoded, agent_mask)
        agent_fuse = self.fuse(torch.cat([agent_agent, map_encoded],1), global_mask)

        interaction_feature = agent_agent[:,0]
        ego_fuse = agent_fuse[:,0] + interaction_feature

        # task = torch.cat([w, goal], -1)
        # task_feature = self.preference(task)
        task_feature = self.preference(goal)

        output = torch.cat([ego_fuse, task_feature], -1)
        return output

class FeatureNet_policy(nn.Module):
    def __init__(self, 
                 action_dim,
                 for_value=False):#use_interaction):
        super(FeatureNet_policy, self).__init__()

        self.for_value = for_value
        self.ation_dim = action_dim

        # agent/map layer
        self.agent_net= AgentEncoder()
        self.map_net = MapEncoder()

        # self.bev_net =CNNExtractor()
        
        # attention layers
        self.agent_agent = Agent2Agent()
        #self.map2agent = CrossTransformer()
        self.fuse = SelfTransformer()

        self.preference =  nn.Sequential(nn.Linear(2, 128), 
                                         nn.ReLU(), 
                                         nn.Linear(128, 64))


    def forward(self, observations):
        ego = observations['ego_state']
        neighbors = observations['neighbors_state']
        map = observations['ego_map']
        # bev = observations['bev']
        goal = observations['goal']
        w = observations['weight']

        # bev_feature = self.bev_net(bev)
        # mask generation
        agent_mask = torch.eq(torch.sum(torch.cat([neighbors[:,:1], 
                                                   neighbors], 1)[:,:,-1], -1), 0)

        agent_mask[:, 0] = False
        map_mask = torch.eq(torch.sum(torch.cat([map[:,:1], 
                                                 map], 1), (-2,-1)), 0)
        map_mask[:, 0] = False
        #print(map_mask.shape, agent_mask.shape)
        global_mask = torch.cat([agent_mask, map_mask], -1)

        # element-wise features
        agent_encoded = self.agent_net(ego, neighbors, w)
        map_encoded = self.map_net(map, goal)

        agent_agent = self.agent_agent(agent_encoded, agent_mask)
        agent_fuse = self.fuse(torch.cat([agent_agent, map_encoded],1), global_mask)

        interaction_feature = agent_agent[:,0]
        ego_fuse = agent_fuse[:,0] + interaction_feature
    
        # task = torch.cat([w, goal], -1)
        # task_feature = self.preference(task)
        task_feature = self.preference(goal)

        # output = torch.cat([ego_fuse, bev_feature, task_feature], -1)
        # output = torch.cat([th.zeros_like(ego_fuse), bev_feature, task_feature], -1)
        if self.for_value:
            return ego_fuse, #bev_feature, task_feature

        output = torch.cat([ego_fuse, task_feature], -1)
        return output

class QNet(nn.Module):
    def __init__(self, 
                 out_dim: int,
                 random_prior = False,
                 ):
        super(QNet, self).__init__()
        self.random_prior = random_prior
        self.fc = nn.Sequential(nn.Linear(256+64, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512, out_dim))
        
        if random_prior:
            self.fc_p = nn.Sequential(nn.Linear(128+32, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, out_dim))
        
            for param in self.fc_p.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.random_prior:
            return self.fc(x) + self.fc_p(x)
        
        return self.fc(x)

class DeQnets(nn.Module):
    def __init__(
        self,
        action_dim,
        model_type: str,
        share_feature: bool,
        ensemble_size: int = 5,
        random_prior: bool = False,
        reward_dim: int = 2,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        self.reward_dim = reward_dim
        self.share_feature = share_feature
        self.random_prior = random_prior
        self.model_type = model_type
        if model_type == "baseline":
            self.out_dim = 1
        if model_type == "mo":
            self.out_dim = 2
        if model_type == "quantile":
            self.out_dim = 25
        if model_type == "gaussian":
            self.out_dim = 5
        

        self.ensemble_size = ensemble_size
        self.latent_dim_pi = 128 
        
        if share_feature:
            self.feature_extractor_reward = FeatureNet_td3(use_action=True, action_dim=action_dim).to(device)
            self.feature_extractor_cost = FeatureNet_td3(use_action=True, action_dim=action_dim).to(device)
            #self.norm = nn.LayerNorm(256+64).to(device)
        else:
            self.feature_extractors = nn.ModuleList([FeatureNet_td3(use_action=True, action_dim=action_dim).to(device) for i in range(self.ensemble_size)])
            #self.norms = nn.ModuleList([nn.LayerNorm(256+64).to(device) for i in range(self.ensemble_size)])
        
        self.qnets = nn.ModuleList([QNet(self.out_dim, self.random_prior).to(device) 
                           for i in range(self.ensemble_size)])

    def forward(self, obs, action) -> th.Tensor:
        Qs = []
        if self.share_feature:
            h_1 = self.feature_extractor_reward(obs, action)
            h_2 = self.feature_extractor_cost(obs, action)
            if self.model_type == "baseline":
                split_num = 2
            else:
                split_num = self.ensemble_size//2

            for i in range(self.ensemble_size):
                if i < split_num:
                    Q = self.qnets[i](h_1)
                else:
                    Q = self.qnets[i](h_2)
                Qs.append(Q)

        else:
            for i in range(self.ensemble_size):
                h = self.feature_extractors[i](obs, action)
                #h = self.norms[i](h)
                Q = self.qnets[i](h)
                Qs.append(Q)

        Qs = th.stack(Qs, -1) # (B,d,N)
        if self.model_type == "moq":
            Qs = Qs.unflatten(dim=-2, sizes=(2, 25))
        return Qs
    
    def set_training_mode(self, training: bool):
        if training:
            self.train()  # Switch to training mode
        else:
            self.eval()

class ActorNet(nn.Module):
    def __init__(
        self,
        action_dim,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        self.latent_dim_pi = 256 
        
        self.feature_extractor = FeatureNet_policy(action_dim=action_dim).to(device)
        self.actionnets = ActionNet(action_dim)

    def forward(self, obs) -> th.Tensor:
        h = self.feature_extractor(obs)
        action = self.actionnets(h)

        return action


class Actor(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor = None,
        #features_dim: int,
        activation_fn = None,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        #self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        self.actor_net = ActorNet(action_dim)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic=False) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        #actions = self.extract_features(obs, self.features_extractor)
        actions = self.actor_net(obs)

        mean, log_std = th.split(actions, 2, -1)
        log_std = th.clamp(log_std, -20., 2.)
        return self.action_dist.actions_from_params(0.01*mean, log_std, deterministic=deterministic)
    
    def raw_forward(self, obs):
        actions = self.actor_net(obs)

        mean, log_std = th.split(actions, 2, -1)
        log_std = th.clamp(log_std, -20., 2.)
        return 0.01*mean, log_std.exp()

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation, deterministic)
    
    def action_log_prob(self, obs):
        actions = self.actor_net(obs)

        mean, log_std = th.split(actions, 2, -1)
        log_std = th.clamp(log_std, -20., 2.)

        return self.action_dist.log_prob_from_params(0.01*mean, log_std)
    
    def set_training_mode(self, training: bool):
        if training:
            self.actor_net.train()  # Switch to training mode
        else:
            self.actor_net.eval()   # Switch to evaluation mode
    
class TD3Policy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        features_extractor_class,
        model_type: str,
        share_feature: bool,
        ensemble_size: int = 5,
        random_prior: bool = False,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class= th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        self.action_dim = action_space.shape[0]

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            #"activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor
        self.observation_space = observation_space
        self.action_space = action_space

        self.share_feature = share_feature
        self.random_prior = random_prior
        self.model_type = model_type
        self.ensemble_size = ensemble_size

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )


        self.critic = DeQnets(self.action_dim,
                                self.model_type,
                                self.share_feature,
                                self.ensemble_size,
                                self.random_prior)
        self.critic_target = DeQnets(self.action_dim,
                                     self.model_type,
                                     self.share_feature,
                                     self.ensemble_size,
                                     self.random_prior)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        #actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(
            observation_space=self.observation_space,
            action_space=self.action_space,
        ).to(self.device)

    def forward(self, observation, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


def combined_extractor_td3(config):
    kwargs = {}
    kwargs["policy"] = TD3Policy
    kwargs["policy_kwargs"] = dict()
    kwargs.update(config.get("alg", {}))

    return kwargs
