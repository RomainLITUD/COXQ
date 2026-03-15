import os

os.environ["CUDA_VISIBLE_DEVICES"]= "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MKL_NUM_THREADS"]      = "20"
os.environ["NUMEXPR_NUM_THREADS"]  = "20"
os.environ["OMP_NUM_THREADS"]      = "20"


import sys
from pathlib import Path

# Required to load inference module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
#print(sys.path)
#sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import warnings
from datetime import datetime
#from itertools import cycle, islice
from typing import Any, Dict
#from decimal import Decimal
#import inspect
import numpy as np
import gymnasium as gym
from stable_baselines3.common.save_util import load_from_zip_file
# Load inference module to register agent
#import stable_baselines3 as sb3lib
import torch as th
import yaml
from contrib_policy import network
from contrib_policy import policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
#from torchinfo import summary
from train.env import make_env, make_parallel_env
from train.utils import ObjDict
from train.metd3 import DETD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from functools import partial
from smarts.env.gymnasium.wrappers.parallel_env import ParallelEnv
from smarts.core.agent_interface import RGB


from smarts.zoo import registry
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

from smarts.core.agent_interface import AgentInterface, RoadWaypoints, EventConfiguration
from smarts.core.controllers import ActionSpaceType

print("\n")
print(f"Torch cuda is available: {th.cuda.is_available()}")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Torch device: {device}")
print("\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# pytype: disable=attribute-error

def load_config():
    """Load config file."""
    parent_dir = Path(__file__).resolve().parent
    config_file = yaml.safe_load((parent_dir / args.config).read_text())
    config = ObjDict(config_file["smarts"])
    return config

def get_nopass():
    nopass_lanes = [
        [":junction-intersection_0_0", ":junction-intersection_4_0", 
         ":junction-intersection_4_1", ":junction-intersection_11_0"],
    ]
    return nopass_lanes

def main(args: argparse.Namespace):
    parent_dir = Path(__file__).resolve().parent
    config = load_config()
    nopass_lanes = get_nopass()

    agent_interface = AgentInterface(
        action = ActionSpaceType.RelativeTargetPose,
        road_waypoints = RoadWaypoints(horizon=50), 
        lidar_point_cloud=False,
        event_configuration = EventConfiguration(),
        max_episode_steps=250,
        # top_down_rgb = False,
        #top_down_rgb = RGB(
        #    width=200,
        #    height=200,
        #    resolution=100 / 200, 
        #),
    )
    
    agent_spec = AgentSpec(
        interface=agent_interface,
        agent_builder=policy.Policy,
    )

    def entry_point(**kwargs):
        return agent_spec
    

    # Register the agent.
    register(
        "custom-policy-v0",
        entry_point=entry_point,
    )

    # Load env config.
    config.mode = args.mode
    config.head = args.head

    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        logdir = parent_dir / "logs" / time
    else:
        logdir = parent_dir / "logs" / args.logdir #Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = logdir
    print("\nLogdir:", logdir, "\n")

    # Setup model.
    if config.mode == "evaluate":
        # Begin evaluation.
        config.model = args.model
        print("\nModel:", config.model, "\n")
    elif config.mode == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f"Expected 'train' or 'evaluate', but got {config.mode}.")

    # Make agent specification
    agent_spec = registry.make(locator=config.agent_locator)

    scenario_path = []
    for scenario in config.scenarios:
        scenario_path.append(str(Path(__file__).resolve().parents[3] / scenario))

    print(len(scenario_path), " training scenarios")
    if config.mode == 'train':
        envs_train = SubprocVecEnv([make_parallel_env(
                                env_id=config.env_id,
                            scenario=scenario_path[0],
                            agent_spec=agent_spec,
                            config=config,
                            seed=config.seed,
                            weights = 0,
                            forbidden= nopass_lanes[0], worker_id = i
                        ) for i in range(config.num_env)])

    envs_eval = SubprocVecEnv([make_parallel_env(
                                env_id=config.env_id,
                            scenario=scenario_path[0],
                            agent_spec=agent_spec,
                            config=config,
                            seed=config.seed,
                            weights = 0,
                            train_=False,
                            forbidden= nopass_lanes[0], worker_id = j + 300
                        ) for j in range(len(scenario_path))])
    
    #print(type(envs_train))
    '''
    envs_train = make_vec_env(make_parallel_env(
                                env_id=config.env_id,
                            scenario=scenario_path[0],
                            agent_spec=agent_spec,
                            config=config,
                            seed=config.seed,
                            weights = 0,
                            train_=False,
                            forbidden= nopass_lanes[0]
                        ), n_envs=1)
    
    envs_eval = make_vec_env(make_parallel_env(
                                env_id=config.env_id,
                            scenario=scenario_path[0],
                            agent_spec=agent_spec,
                            config=config,
                            seed=config.seed,
                            weights = 0,
                            train_=False,
                            forbidden= nopass_lanes[0]
                        ), n_envs=1)
    '''   
    # Run training or evaluation.
    print('start...')
    if config.mode == "train":
        train(
            envs_train=envs_train,
            envs_eval=envs_eval,
            config=config,
            agent_spec=agent_spec,
            parent_dir = parent_dir
        )
    else:
        evaluate(envs=envs_eval, config=config, agent_spec=agent_spec, parent_dir = parent_dir)

    # Close all environments
    for env in envs_train.values():
        env.close()
    for env in envs_eval.values():
        env.close()


def train(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
    agent_spec: AgentSpec,
    parent_dir
):
    print("\nStart training.\n")
    save_dir = config.logdir / "train"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=config.logdir / "checkpoint",
        name_prefix="TD3",
    )

    #scenarios_iter = islice(cycle(config.scenarios), config.epochs)
    #for index, scen in enumerate(scenarios_iter):
        

    model = DETD3(
        env=envs_train,
        tensorboard_log=config.logdir / "tensorboard",
        verbose=1,
        **network.combined_extractor_td3(config),
    )

    #parent_dir = Path(__file__).resolve().parent
    #model = DEPPO.load(save_dir / "intermediate")
   #  model = DETD3.load(parent_dir / "logs" / "sac_lt" / "checkpoint" / "TD3_512000_steps", 
   #                      envs_train,**network.combined_extractor_td3(config))

    eval_callback = EvalCallback(
        envs_eval,
        best_model_save_path=config.logdir / "eval",
        n_eval_episodes=config.eval_eps,
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model.set_env(envs_train)
    model.learn(
        total_timesteps=config.train_steps,
        log_interval=config.log_interval,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False,
    )
    model.save(save_dir / "intermediate")

    print("Finished training.")

    # Save trained model.
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model.save(save_dir / ("model_" + time))
    print("\nSaved trained model.\n")

    np.savez("coxq_tj.npz", rewards=model.replay_buffer.rewards_nd)

def evaluate(
    envs: Dict[str, gym.Env],
    config: Dict[str, Any],
    agent_spec: AgentSpec,
    parent_dir
):
    print("\nEvaluate policy.\n")
    device = th.device("cpu")
    
    ckpt = str(parent_dir / "logs" / "coxq_ot" / "checkpoint" / "TD3_512000_steps.zip")
    model = DETD3(
        env=envs,
        tensorboard_log=config.logdir / "tensorboard",
        verbose=1,
        **network.combined_extractor_td3(config),
    )
    model.set_parameters(ckpt, exact_match=False, device=device)
    # model.set_env(envs)
    mean_reward, mean_return, mean_cost, mean_lenth  = evaluate_policy(
            model, envs, n_eval_episodes=2000, deterministic=True, return_episode_rewards = True
    )
    
    print(np.array(mean_reward).mean(), np.array(mean_return).mean(), np.array(mean_cost).mean(), np.array(mean_lenth).mean())
    # print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")
    print("\nFinished evaluating.\n")


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path for saving logs.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--config",
        help="configuration file .yaml",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--head", help="Display the simulation in Envision.", action="store_true"
    )

    args = parser.parse_args()

   # if args.mode == "evaluate" and args.model is None:
    #    raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)
