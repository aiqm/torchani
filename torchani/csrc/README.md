# CSRC
Cpp source files for CUAEV and MNP extensions.
- CUAEV: CUDA Extension for AEV calculation. Performance improvement is
  expected to be ~3X for AEV computation and ~1.5X for energy training, 2.6X
  for energy+force training.
- MNP: Multi Net Parallel between different species networks using multithreading
  (Inference Only) to reduce CUDA call overhead.

## Requirements

Install a PyTorch version with CUDA support, and a CUDA Toolkit appropriate for that
PyTorch version, for example:

```bash
pip install torch==2.13 --index-url https://download.pytorch.org/whl/cu130
conda install nvidia::cuda-toolkit=13.0
```

## Build from source

```bash
# Clone the repo and cd to the directory
git clone https://github.com/aiqm/torchani.git
cd ./torchani

# Create a conda (or mamba) environment with the optional dependencies
# (alternatively use a venv and run 'pip install -r dev_requirements.txt')
conda env create -f ./environment.yaml
conda activate ani

# Install torchani
pip install --no-deps -v -e .

# Build the extensions, for all detected SMs
ani build-extensions
# Or for specific SMs
ani build-extensions --sm 8.0 --sm 8.9
# Or for all SMs supported by PyTorch
ani build-extensions --all-sms
```

## Update

```bash
cd torchani
git pull
pip install --no-deps -v -e .
ani build-extensions
```

To build the extensions in an HPC cluster (access to conda/mamba either
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
# Building with --all-sms guarantees that the extensions will run correctly
# whatever the GPU is you select during runs, as long as pytorch supports that GPU
git clone https://github.com/aiqm/torchani.git
cd ./torchani
conda env create -f ./environment.yaml
conda activate ani
pip install --no-deps -v -e .
ani build-extensions --all-sms
```

## Test
```bash
cd torchani
pip install -r dev_requirements.txt
./download-dev-data.sh
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
python tools/training-aev-benchmark.py dev-data/hf-data/ani-1x/sample.h5
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
