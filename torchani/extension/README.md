# CUAEV
CUDA Extension for AEV calculation.


### Install
In most cases, if gcc and cuda environment are well configured, runing the following command will install the cuaev package.
```bash
pip install -e .
```

Notes for install on Hipergator
```bash
srun -p gpu --gpus=geforce:1 --time=01:00:00 --mem=10gb  --pty -u bash -i  # compile may fail because of low on memery (when memery less than 5gb)
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch              # make sure it's cudatoolkit=10.0
module load cuda/10.0.130
module load gcc/7.3.0
pip install -e .
```


### Limitations
Current implementation of CUAEV does not support pbc and force calculation.