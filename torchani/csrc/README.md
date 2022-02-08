# CSRC
Cpp source files for CUAEV and MNP extensions.
- CUAEV: CUDA Extension for AEV calculation. Performance improvement is expected to be ~3X for AEV computation and ~1.5X for energy training, 2.6X for energy+force training.
- MNP: Multi Net Parallel between different species networks using OpenMP (Inference Only) to reduce CUDA call overhead.

## Requirement
The extensions need the nightly version [pytorch](https://pytorch.org/) to be able to work.
If you use conda and your cuda version is 11.1, you could install it by
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
```

## Build from source
In most cases, if `gcc` and `cuda` environment are well configured, runing the following command at `torchani` directory will install torchani and all extensions together.

```bash
git clone git@github.com:roitberg-group/torchani_sandbox.git
cd torchani
# choose one option below
# use --ext-all-sms if you are building in SLURM environment and there are multiple different gpus in a node
# use --ext will only build for detected gpus
python setup.py install --ext          # only build for detected gpus
python setup.py install --ext-all-sms  # build for all gpus
# or for development
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

Some notes for building extensions on multiple HPC
<details>
<summary>Bridges2</summary>

```bash
# prepare
srun -p GPU-small --ntasks=1 --cpus-per-task=5 --gpus=1 --time=02:00:00 --mem=20gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# modules
module load cuda/10.2.0
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
# install torchani
git clone https://github.com/roitberg-group/torchani_sandbox.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--ext"
```

</details>

<details>
<summary>Hipergator</summary>

```bash
srun -p gpu --ntasks=1 --cpus-per-task=2 --gpus=geforce:1 --time=02:00:00 --mem=10gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# modules
module load cuda/11.1.0 gcc/7.3.0 git
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
# install torchani
git clone https://github.com/roitberg-group/torchani_sandbox.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--ext"
```

</details>

<details>
<summary>Expanse</summary>

```bash
srun -p gpu-shared --ntasks=1 --account=cwr109 --cpus-per-task=1 --gpus=1 --time=01:00:00 --mem=10gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# modules
module load cuda10.2/toolkit/10.2.89 gcc/7.5.0
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
# install torchani
git clone https://github.com/roitberg-group/torchani_sandbox.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--ext"
```

</details>


<details>
<summary>Moria</summary>

```bash
srun --ntasks=1 --cpus-per-task=2 --gres=gpus:1 --time=02:00:00 --mem=10gb  --pty -u bash -i
# create env if necessary
conda create -n cuaev python=3.8
conda activate cuaev
# cuda path (could be added to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
# install torchani
git clone https://github.com/roitberg-group/torchani_sandbox.git
cd torchani
pip install -e . && pip install -v -e . --global-option="--ext-all-sms"
```

</details>

## Test
```bash
cd torchani
pip install pytest pynvml pkbar ase parameterized h5py expecttest
./download.sh
# cuaev
python tests/test_cuaev.py
# mnp
python tests/test_infer.py
```

## Usage
#### CUAEV
Pass `use_cuda_extension=True` when construct aev_computer, for example:
```python
cuaev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)
# or
cuaev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=True)
```

#### MNP
```python
ani2x = torchani.models.ANI2x()
# ensemble
bmm_ensemble = ani2x.neural_networks.to_infer_model(use_mnp=True)
# single model
model = ani2x.neural_networks[0].to_infer_model(use_mnp=True)
```

## TODOs
- [x] CUAEV Forward
- [x] CUAEV Backwad (Force)
- [x] CUAEV Double Backwad (Force training need aev's double backward w.r.t. grad_aev)
- [ ] PBC

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
```
pip install pynvml pkbar
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
