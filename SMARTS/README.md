# SMARTS

For SMARTS, the main body of the COX-Q code is in `coxq/train/metd3.py`. Well, the file name is TD3 but actually SAC.

## Preparation

For now, the code only supports `python==3.8/3.9`
- Install SMARTS following the instructions [here](https://smarts.readthedocs.io/en/latest/setup.html).
- Set up the *Driving SMARTS 2023.1 & 2023.2* environment following the instructions [here](https://smarts.readthedocs.io/en/latest/benchmarks/driving_smarts_2023_1.html).
- Put the provided `coxq` folder in `SMARTS\example\`.

- TO ADD stable_baselines3 modification details

## Training

There are two options: run the code in Docker following the instructions [here](https://smarts.readthedocs.io/en/latest/benchmarks/driving_smarts_2023_1.html#docker), or just activate the virtual environment and run:
```
python3.8 train/run.py
```
