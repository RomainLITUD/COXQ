from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
import warnings
import torch

# nohup bash -c "python -u onpolicy_benchmark.py" > output0.log 2> error0.log

if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Onpolicy_PP1')

    # set up the algorithms.
    on_policys = ['TRPOPID', 
                  'RCPO', 
                  'PPOSimmerPID', 
                  'CUP'
                  ]
    eg.add(
        'algo',
        on_policys
    )

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # the default configs here are as follows:
    # eg.add('algo_cfgs:steps_per_epoch', [20000])
    # eg.add('train_cfgs:total_steps', [20000 * 500])
    # which can reproduce results of 1e7 steps.

    # if you want to reproduce results of 1e6 steps, using
    eg.add('algo_cfgs:steps_per_epoch', [2000])
    eg.add('train_cfgs:total_steps', [2000 * 600])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('algo_cfgs:gamma', [0.975])
    eg.add('algo_cfgs:cost_gamma', [0.975])

    # set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    # if you want to use GPU, please set gpu_id like follows:
    # gpu_id = [0, 1, 2, 3]
    # if you want to use CPU, please set gpu_id = None
    # we recommends using CPU to obtain results as consistent
    # as possible with our publicly available results,
    # since the performance of all on-policy algorithms
    # in OmniSafe is tested on CPU.
    gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    # set up the environment.
    eg.add('env_id', [
        'SafetyPointButton2-v0',
        'SafetyPointGoal2-v0',
        'SafetyCarButton1-v0',
        'SafetyCarButton2-v0',
        'SafetyPointPush1-v0',
        ])
    eg.add('seed', [5*i for i in range(10)])

    # total experiment num must can be divided by num_pool.
    # meanwhile, users should decide this value according to their machine.
    eg.run(train, num_pool=50, gpu_id=gpu_id)
