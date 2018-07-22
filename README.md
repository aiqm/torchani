# TorchANI

[![Codefresh build status]( https://g.codefresh.io/api/badges/build?repoOwner=zasdfgbnm&repoName=torchani&branch=master&pipelineName=torchani&accountName=zasdfgbnm&key=eyJhbGciOiJIUzI1NiJ9.NTk5ZmEwNzI2MTNhNTMwMDAxNTY4MmJm.nnVU1i-VQQSzPcsGxKnMC0wT-y9C2i8xuBZvUjlubYg&type=cf-1)]( https://g.codefresh.io/repositories/zasdfgbnm/torchani/builds?filter=trigger:build;branch:master;service:5b53d92fff9e565ae1f3a5b5~torchani)

TorchANI is a pytorch implementation of ANI.

Paper:

* Smith JS, Isayev O, Roitberg AE. ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. Chemical science. 2017;8(4):3192-203.

# Install

TorchANI requires the master branch of PyTorch, which means:

* You need to compile install the latest pytorch, see [official instructions](https://github.com/pytorch/pytorch#from-source).
* Some update to TorchANI might require the user to recompile install the latest PyTorch. Before submitting a bug report, make sure you are running the latest PyTorch.

After installing the correct PyTorch, all you need is clone the repository and do:

```bash
pip install .
```

# Development

Never commit to the master branch directly. If you need to change something, create a new branch, submit a PR on GitHub.

You must pass all the tests on GitHub before your PR can be merged.

Code review is required before merging pull request.

To manually run unit tests, do `python setup.py test`