# TorchANI

[![Codefresh build status]( https://g.codefresh.io/api/badges/build?repoOwner=zasdfgbnm&repoName=torchani&branch=master&pipelineName=torchani&accountName=zasdfgbnm&key=eyJhbGciOiJIUzI1NiJ9.NTk5ZmEwNzI2MTNhNTMwMDAxNTY4MmJm.nnVU1i-VQQSzPcsGxKnMC0wT-y9C2i8xuBZvUjlubYg&type=cf-1)]( https://g.codefresh.io/repositories/zasdfgbnm/torchani/builds?filter=trigger:build;branch:master;service:5b53d92fff9e565ae1f3a5b5~torchani)

TorchANI is a pytorch implementation of ANI. It is currently under alpha release, which means, the API is not stable yet. If you find a bug of TorchANI, or have some feature request, feel free to open an issue on GitHub, or send us a pull request.

# Install

TorchANI requires the master branch of PyTorch, which means:

You need to install the latest preview version of pytorch\
For install
```bash
conda install pytorch-nightly -c pytorch
```
For update
```bash
conda update pytorch-nightly
``` 

After installing the correct PyTorch, all you need is clone the repository and do:

```bash
pip install .
```

After TorchANI has been installed, you can build the documents by running `sphinx-build docs build`. But make sure you
install dependencies:
```bash
pip install sphinx sphinx-gallery pillow matplotlib sphinx_rtd_theme
```

# Paper

The original ANI-1 paper is:

* Smith JS, Isayev O, Roitberg AE. ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. Chemical science. 2017;8(4):3192-203.

We are planning a seperate paper for TorchANI, it will be available when we are ready for beta release of TorchANI.

# Note to TorchANI developers

Never commit to the master branch directly. If you need to change something, create a new branch, submit a PR on GitHub.

You must pass all the tests on GitHub before your PR can be merged.

Code review is required before merging pull request.

To manually run unit tests, do `python setup.py test`
