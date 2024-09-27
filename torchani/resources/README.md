# Constants and parameter files

Contains constants and parameter files packaged directly into torchani and
needed to compute various potentials. Data is stored either in `HDF5` or `JSON`
format. This data is not meant to be directly modified or accessed, we consider
it an implementation detail how the data is stored and managed by TorchANI.
Users of the library should access the data through the `torchani.constants`
module. If you use these constants in your work please cite the corresponding
article(s).
