# CUAEV
CUDA Extension for AEV calculation.
Performance improvement is expected to be ~3X for AEV computation and ~1.5X for overall training workflow.

## Install
In most cases, if `gcc` and `cuda` environment are well configured, runing the following command at `torchani` directory will install torchani and cuaev together.
```bash
git clone git@github.com:aiqm/torchani.git
cd torchani
# install by
python setup.py install --cuaev
# or for development
# `pip install -e . && ` is only needed for the very first install (because issue of https://github.com/pypa/pip/issues/1883)
pip install -e . && pip install -e . --global-option="--cuaev"
```

<del>Notes for install on Hipergator</del> (Currently not working because Pytorch dropped the official build for cuda/10.0)
```bash
srun -p gpu --gpus=geforce:1 --time=01:00:00 --mem=10gb --pty -u bash -i   # compile may fail because of low on memery (when memery is less than 5gb)
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch              # make sure it's cudatoolkit=10.0
module load cuda/10.0.130
module load gcc/7.3.0
python setup.py install --cuaev
```

## Usage
Pass `use_cuda_extension=True` when construct aev_computer, for example:
```python
cuaev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)
```

## TODOs
- [x] CUAEV Forward
- [x] CUAEV Backwad (Force)
- [ ] PBC
- [ ] Force training (Need cuaev's second derivative)

## Benchmark
Benchmark of [torchani/tools/training-aev-benchmark.py](https://github.com/aiqm/torchani/tree/master/torchani/tools/training-aev-benchmark.py) on TITAN V:

| ANI-1x dataset (Batchsize 2560) | Energy Training         | Energy and Force Inference        |
|---------------------------------|-------------------------|-----------------------------------|
| Time per Epoch / Memory         | AEV / Total / GPU Mem   |  AEV  / Force / Total / GPU Mem   |
| aev cuda extension              | 3.90s / 31.5s / 2088 MB | 3.90s / 22.6s / 43.0s / 4234 MB   |
| aev python code                 | 23.7s / 50.2s / 3540 MB | 25.3s / 48.0s / 88.2s / 11316 MB  |

## Test
```bash
cd torchani
./download.sh
python tests/test_cuaev.py
```

benchmark
```
python tools/training-aev-benchmark.py download/dataset/ani-1x/sample.h5
python tools/aev-benchmark-size.py
```