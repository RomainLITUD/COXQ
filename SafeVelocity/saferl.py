import functools
import random
import jax

import numpy as np
import brax
from sac import train as sac
from brax.io import model
import argparse

import os
os.environ["JAX_DETERMINISTIC"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameters")
    # ---------- algorithm hyper-params --------------------------------------
    parser.add_argument("--env", default="humanoid",
                        help="Brax environment name")
    
    parser.add_argument("--num_obj", type=int, default=1,
                        help="number of objectives")
    
    parser.add_argument("--save_policy", type=bool, default=False,
                        help="save policy params")
    
    parser.add_argument("--model_mode", default="qr",
                        help="Model mode")
    
    parser.add_argument("--learning_method", default="tqc",
                        help="learning method")
    
    parser.add_argument("--explore_method", default="coxq",
                        help="exploration method")
    
    parser.add_argument("--exploration", type=bool, default=True,
                        help="exploration mode on or off")

    parser.add_argument("--save_file", default="name",
                        help="name of saved file")
    
    parser.add_argument("--ensemble_size", type=int, default=5,
                        help="ensemble size")
    
    parser.add_argument("--tail_reward", type=int, default=2,
                        help="tail reward drop")
    
    parser.add_argument("--tail_cost", type=int, default=5,
                        help="tail cost drop")
    
    parser.add_argument("--convex", type=float, default=10.,
                        help="convex coefficient")
    
    parser.add_argument("--env_steps", type=int, default=3072000,
                        help="Training timesteps")
    
    parser.add_argument("--nb_layers", type=int, default=2,
                        help="Number of layers for critic network")
    
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden size of layers for critic network")
    
    parser.add_argument("--beta_reward", type=float, default=4.,
                        help="exploration step and direction coefficient for reward")
    
    parser.add_argument("--beta_cost", type=float, default=3.,
                        help="exploration step and direction coefficient for cost")

    parser.add_argument("--budget", type=float, default=2.5,
                        help="exploration step and direction coefficient")
    
    parser.add_argument("--budget_st", type=float, default=2.5,
                        help="budget single step")

    parser.add_argument("--gpu_id", default="1",
                        help="gpu_id")

    return parser.parse_args()


def main():
    # constraints = {'hopper': 74.02,
    #                'walker2d': 234.15,
    #                'ant': 262.22,
    #                'humanoid': 141.19}    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 
    random.seed(1)
    np.random.seed(1)

    N = 10
    random_integers = [5*i for i in range(N)]

    train_fn = functools.partial(sac.train, num_timesteps=args.env_steps, num_evals=101, 
                  episode_length=1000, normalize_observations=True, action_repeat=1, actor_delay=1,
                  discounting=0.99, learning_rate=3e-4, q_learning_rate=3e-4, num_envs=64, 
                  batch_size=256, num_eval_envs=10, multiplier_learning_rate=1e-5,
                  grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=args.env_steps, 
                  min_replay_size=10240, network_size = (args.hidden_size,)*args.nb_layers, 
                  )

    X, Y, R = [], [], []
    for i in range(N):
        print(i, random_integers[i])
        env = brax.envs.get_environment(args.env, backend = "generalized")
        x, y, r_s, params = train_fn(environment=env,
                                     name=args.env, 
                                     mode=args.model_mode,
                                     num_obj=args.num_obj,
                                    ensemble_size=args.ensemble_size,
                                    seed = random_integers[i],
                                    method = args.learning_method,
                                    tail_r = args.tail_reward,
                                    tail_c = args.tail_cost,
                                    cost_limit = 2.5,
                                    budget = args.budget,
                                    budget_st = args.budget_st,
                                    convex_coeff= args.convex,
                                    topk = 13,
                                    beta_reward = args.beta_reward,
                                    beta_cost = args.beta_cost,
                                    exploration_strategy = args.exploration,
                                    exploration_method = args.explore_method,
                                    delta = 4.,
                                    )
        if args.save_policy:
            model.save_params('./policy/' + args.env + '/' +args.save_file, params)
        
        jax.clear_caches()
        X.append(x)
        Y.append(y)
        R.append(r_s)

    X = np.array(X)
    Y = np.array(Y)
    R = np.array(R)

    print(Y.shape)
    print(R.shape)

    np.savez('./boundary/' + args.env + '/' +args.save_file, x = X, y = Y, rewards=R)

if __name__ == "__main__":
    main()
