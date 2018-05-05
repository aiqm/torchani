from benchmark import Benchmark


class NeighborAEVBenchmark(Benchmark):

    def oneByOne(self, coordinates, species):
        """Benchmarking the given dataset of computing energies and forces one at a time

        Parameters
        ----------
        coordinates : numpy.ndarray
            Array of shape (conformations, atoms, 3)
        species : list
            List of species for this molecule. The length of the list must be the same as
            atoms in the molecule.

        Returns
        -------
        dict
            Dictionary storing the times for computing AEVs, energies and forces, in milliseconds.
            The dictionary should contain the following keys:
            aev : the time used to compute AEVs from coordinates.
            energy : the time used to compute energies, when the AEVs are given.
            force : the time used to compute forces, when the energies and AEVs are given.
        """
        # return { 'aev': 0, 'energy': 0, 'force': 0 }
        raise NotImplementedError('subclass must implement this method')

    def inBatch(self, coordinates, species):
        """Benchmarking the given dataset of computing energies and forces in batch mode

        The signature of this function is the same as `oneByOne`"""
        # return { 'aev': 0, 'energy': 0, 'force': 0 }
        raise NotImplementedError('subclass must implement this method')
