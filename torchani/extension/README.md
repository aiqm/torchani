# CUAEV
CUDA Extension for AEV calculation.
Performance improvement is expected to be ~3X for AEV computation and ~1.5X for overall training workflow.

## Install
In most cases, if `gcc` and `cuda` environment are well configured, runing the following command at `torchani/extension` directory will install the cuaev package.
```bash
pip install -e .
```

Notes for install on Hipergator
```bash
srun -p gpu --gpus=geforce:1 --time=01:00:00 --mem=10gb --pty -u bash -i   # compile may fail because of low on memery (when memery is less than 5gb)
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch              # make sure it's cudatoolkit=10.0
module load cuda/10.0.130
module load gcc/7.3.0
pip install -e .
```

## Usage
Pass `use_cuda_extension=True` when construct aev_computer, for example:
```
cuaev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)
```

## Limitations
Current implementation of CUAEV does not support pbc and force calculation.

## Benchmark
Benchmark of [torchani/tools/training-aev-benchmark.py](https://github.com/aiqm/torchani/tree/master/torchani/tools/training-aev-benchmark.py) on RTX 2080 Ti:

|         ANI-1x          |     Without Shuffle     |         Shuffle         |
|:-----------------------:|:-----------------------:|:-----------------------:|
| Time per Epoch / Memory |  AEV / Total / GPU Mem  |  AEV / Total/ GPU Mem   |
|   aev cuda extension    | 7.7s  / 26.3s / 2289 MB | 8.5s / 27.6s / 2425 MB  |
|     aev python code     | 21.1s / 40.0s / 7361 MB | 28.7s / 47.8s / 3475 MB |
|      improvements       |   2.74 / 1.52 / 3.22    |   3.38 / 1.73 / 1.43    |
