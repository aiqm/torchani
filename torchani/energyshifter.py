from .env import buildin_sae_file


class EnergyShifter:
    """Class that deal with self atomic energies.

    Attributes
    ----------
    self_energies : dict
        The dictionary that stores self energies of species.
    """

    def __init__(self, self_energy_file=buildin_sae_file):
        # load self energies
        self.self_energies = {}
        with open(self_energy_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0].split(',')[0].strip()
                    value = float(line[1])
                    self.self_energies[name] = value
                except Exception:
                    pass  # ignore unrecognizable line

    def subtract_sae(self, energies, species):
        """Subtract self atomic energies from `energies`.

        Parameters
        ----------
        energies : pytorch tensor of `dtype`
            The tensor of any shape that stores the raw energies.
        species : list of str
            The list specifying the species of each atom. The length of the
            list must be the same as the number of atoms.

        Returns
        -------
        pytorch tensor of `dtype`
            The tensor of the same shape as `energies` that stores the energies
            with self atomic energies subtracted.
        """
        s = 0
        for i in species:
            s += self.self_energies[i]
        return energies - s

    def add_sae(self, energies, species):
        """Add self atomic energies to `energies`

        Parameters
        ----------
        energies : pytorch tensor of `dtype`
            The tensor of any shape that stores the energies excluding self
            atomic energies.
        species : list of str
            The list specifying the species of each atom. The length of the
            list must be the same as the number of atoms.

        Returns
        -------
        pytorch tensor of `dtype`
            The tensor of the same shape as `energies` that stores the raw
            energies, i.e. the energy including self atomic energies.
        """
        s = 0
        for i in species:
            s += self.self_energies[i]
        return energies + s

    def dataset_subtract_sae(self, data):
        """Allow object of this class to be used as transforms of pytorch's
        dataset.
        """
        data['energies'] = self.subtract_sae(data['energies'], data['species'])
        return data
