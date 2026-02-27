from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
import warnings
import torch

if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='COX-Q')

    # set up the algorithms.
    off_policy = [#'SACUCB', 
                  'BAE', # This is COX-Q
                  #'CAL',
                  #'WCSAC',
                  #'ORAC',
                 ]
    eg.add('algo', off_policy)

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # eg.add('algo_cfgs:convex', [10.])
    # eg.add('algo_cfgs:update_iters', [1])
    #eg.add('algo_cfgs:policy_delay', [10])
    # the default configs here are as follows:
    eg.add('algo_cfgs:steps_per_epoch', [2000])
    eg.add('train_cfgs:total_steps', [2000 * 600])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('lagrange_cfgs:cost_limit', [10.])
    eg.add('lagrange_cfgs:lagrangian_multiplier_init', [0.001])
    
    # eg.add('lagrange_cfgs:lambda_lr', [5e-4])
    # eg.add('lagrange_cfgs:lagrangian_multiplier_init', [1.])
    eg.add('logger_cfgs:window_lens', [1])
    eg.add('algo_cfgs:gamma', [0.975])
    # eg.add('algo_cfgs:cost_gamma', [0.975])
    eg.add('algo_cfgs:warmup_epochs', [0])
    eg.add('algo_cfgs:budget', [10.])
    eg.add('algo_cfgs:tail_reward', [0])
    eg.add('algo_cfgs:tail_cost', [0])
    eg.add('algo_cfgs:alpha', [0.00001])
    eg.add('algo_cfgs:auto_alpha', [False])


    # set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    #gpu_id = [0]
    # if you want to use CPU, please set gpu_id = None
    gpu_id = avaliable_gpus

    # set up the environments.
    eg.add('env_id', [
        'SafetyPointButton2-v0',
        'SafetyPointGoal2-v0',
        'SafetyCarButton1-v0',
        'SafetyCarButton2-v0',
        'SafetyPointPush1-v0',
        ])
    eg.add('seed', [i*5 for i in range(10)])
    eg.run(train, num_pool=50, gpu_id=gpu_id)
