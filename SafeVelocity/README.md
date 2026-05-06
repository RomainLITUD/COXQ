# Safe Velocity Benchmark

We adopted the `safe velocity` tasks from `Safety-Gymnasium` ([here](https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium)) and ported them to Brax to accelerate simulation.

## Preparation

- Install Brax following the instructions [here](https://github.com/google/brax).
- Install Jax[cuda13] following the instructions [here](https://docs.jax.dev/en/latest/installation.html)
- The used environments are provided in `custom_envs/envs`. Users can replace the default brax envs with them in the source code, or directly use `env_construct.py`.

*For now, the code only supports CUDA >= 13.0. Modifications are needed for older JAX/CUDA versions.*

## Training

Run the following command to train across 10 seeds:
```
python -u saferl.py --env=humanoid --env_steps=3072000 --save_file=coxq
```

## Visualization

An easy way of tracking the curves during training or visualizing the final policy is to use the notebook provided by Brax: [here](https://github.com/google/brax/tree/main/notebooks)
