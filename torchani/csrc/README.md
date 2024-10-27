# CSRC
Cpp source files for CUAEV and MNP extensions.
- CUAEV: CUDA Extension for AEV calculation. Performance improvement is
  expected to be ~3X for AEV computation and ~1.5X for energy training, 2.6X
  for energy+force training.
- MNP: Multi Net Parallel between different species networks using OpenMP
  (Inference Only) to reduce CUDA call overhead.

## Requirement
Following [pytorch.org](https://pytorch.org/) to install PyTorch.
On linux, for example:
```
conda install pytorch pytorch-cuda=11.8 cuda-toolkit=11.8 cuda-compiler=11.8 -c pytorch -c nvidia
```

## Build from source

In most cases, if `gcc` and `cuda` environment are well configured, runing the
following command at `torchani` directory will install torchani and all
extensions together.

```bash
git clone git@github.com:roitberg-group/torchani_sandbox.git
cd torchani_sandbox
# choose one option below
# ============== install ==============
python setup.py install --ext          # only build for detected gpus
python setup.py install --ext-all-sms  # build for all gpus
# ============== development ==============

# `pip install -e . && ` is only needed for the very first install (because issue of https://github.com/pypa/pip/issues/1883)
pip install -e . && pip install -v -e . --global-option="--ext"          # only build for detected gpus
pip install -e . && pip install -v -e . --global-option="--ext-all-sms"  # build for all gpus
```

## Update

```bash
cd torchani
git pull
pip install -v -e . --global-option="--ext"
```

To build the extension in a HPC cluster (access to conda/mamba either
by local install or a "module" is needed):

```bash
# First start an interactive session with GPU access. How to do this will
# depend on your specific cluster, here are some examples:
# Hipergator
srun -p gpu --ntasks=1 --cpus-per-task=2 --gpus=geforce:1 --time=02:00:00 --mem=10gb  --pty -u bash -i
# Bridges2
srun -p GPU-small --ntasks=1 --cpus-per-task=5 --gpus=1 --time=02:00:00 --mem=20gb  --pty -u bash -i
# Expanse
srun -p gpu-shared --ntasks=1 --account=cwr109 --cpus-per-task=1 --gpus=1 --time=01:00:00 --mem=10gb  --pty -u bash -i
# Moria
srun --ntasks=1 --cpus-per-task=2 --gres=gpu:1 --time=02:00:00 --mem=10gb  --pty -u bash -i
# Create env and install torchani with the extensions.
# Installing with --ext-all-sms  guarantees that the extension will run correctly
# whatever the GPU is you select during runs, as long as pytorch supports that GPU
conda create -f ./environment.yml
git clone https://github.com/roitberg-group/torchani_sandbox.git
cd torchani_sandbox
pip install  -v --no-deps --no-build-isolation --editable .
pip install -v --no-deps --no-build-isolation --editable . --global-option="--ext-all-sms"

```

## Test
```bash
cd torchani
pip install -r dev_requirements.txt
./download.sh
# cuaev
python tests/test_cuaev.py
# mnp
python tests/test_infer.py
```

## Usage
#### CUAEV

Pass `strategy='cuaev'` or `strategy='cuaev-fused'` when constructing an aev_computer, for example:
```python
cuaev_computer = AEVComputer.from_constants(
    Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, strategy='cuaev',
)
# or
cuaev_computer = AEVComputer.like_1x(cutoff_fn="smooth", strategy='cuaev-fused')
```

#### MNP
```python
ani2x = torchani.models.ANI2x()
# ensemble
bmm_ensemble = ani2x.neural_networks.to_infer_model(use_mnp=True)
# single model
model = ani2x.neural_networks[0].to_infer_model(use_mnp=True)
```

## Benchmark

#### CUAEV
Benchmark of [torchani/tools/training-aev-benchmark.py](https://github.com/roitberg-group/torchani_sandbox/blob/master/tools/training-aev-benchmark.py):

Train ANI-1x dataset (Batchsize 2560) on Tesla V100 for 1 epoch:
```
RUN                Total AEV    Forward      Backward     Force        Optimizer    Others       Epoch time   GPU
0 cu Energy        3.355 sec    4.470 sec    4.685 sec    0.0 ms       3.508 sec    2.223 sec    18.241 sec   2780.8MB
1 py Energy        19.682 sec   4.149 sec    4.663 sec    0.0 ms       3.495 sec    2.220 sec    34.209 sec   4038.8MB
2 cu Energy+Force  3.351 sec    4.200 sec    27.402 sec   16.514 sec   3.467 sec    4.556 sec    59.490 sec   7492.8MB
3 py Energy+Force  19.964 sec   4.176 sec    91.866 sec   36.554 sec   3.473 sec    5.403 sec    161.435 sec  8034.8MB
```

benchmark
```bash
pip install -r dev_requirements.txt
python tools/training-aev-benchmark.py download/dataset/ani-1x/sample.h5
python tools/aev-benchmark-size.py -p
```

For bigger system inference, Also check [#16 CUAEV calculation optimzation](https://github.com/roitberg-group/torchani_sandbox/pull/16)

#### MNP
Check [#35 NN optimization](https://github.com/roitberg-group/torchani_sandbox/pull/35)

```bash
# original
python tools/aev-benchmark-size.py -p
# new
python tools/aev-benchmark-size.py -p -e --infer_model --mnp
```
