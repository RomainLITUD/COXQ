# Safe Navigation

For safe navigation benchmarks, we adopted the original `omnisafe` code: [click](https://github.com/PKU-Alignment/omnisafe/tree/main)

The major modifications include:
- For deep ensembles, we run the models in parallel to accelerate the training and simulations.
- Add some off-policy baselines.
- Add our COX-Q exploration strategy in `omnisafe/adapter`.
- Add the COX-Q method in `omnisafe/algorithms/off_policy/bae.py`.

## Preparation

- `python==3.11, cuda>=12.3`
- Install `omnisafe` following the instructions [here](https://github.com/PKU-Alignment/omnisafe/tree/main).
- An easy way is to replace the original code of `omnisafe` by our provided repository.
- Disable the random layout in `safety-gymnasium`.

Then modify the timesteps in `point.xml` and `car.xml` in `safety-gymnasium/assets/xmls` (same as CVPO paper [here](https://github.com/liuzuxin/cvpo-safe-rl)):
- in `point.xml`, line 19: `<option timestep="0.005"/>`
- in 'car.xml', line 19: `<option timestep="0.01"/>`

## Training

Modify the random seeds and the benchmark models in `offpolicy_benchmark.py`, then run:
```
python -u offpolicy_benchmark.py
```
