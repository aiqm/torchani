from .aev_base import AEVComputer
import torch
import torch.nn as nn
import bz2
import os
import lark
import struct
import copy
import math

class PerSpeciesFromNeuroChem(nn.Module):
    """Subclass of `torch.nn.Module` for the per atom aev->y transformation, loaded from NeuroChem network dir.
    
    Attributes
    ----------
    dtype : torch.dtype
        Pytorch data type for tensors
    layers : int
        Number of layers.
    layerN : torch.nn.Linear
        Linear model for each layer.
    activation : function
        Function for computing the activation for all layers but the last layer.
    activation_index : int
        The NeuroChem index for activation.
    """

    def __init__(self, dtype, filename):
        """Initialize from NeuroChem network directory.

        Parameters
        ----------
        dtype : torch.dtype
            Pytorch data type for tensors
        filename : string
            The file name for the `.nnf` file that store network hyperparameters. The `.bparam` and `.wparam`
            must be in the same directory
        """
        super(PerSpeciesFromNeuroChem, self).__init__()

        self.dtype = dtype
        networ_dir = os.path.dirname(filename)
        with open(filename, 'rb') as f:
            buffer = f.read()
            buffer = self._decompress(buffer)
            layer_setups = self._parse(buffer)
            self._construct(layer_setups, networ_dir)
        

    def _decompress(self, buffer):
        """Decompress the `.nnf` file
        
        Parameters
        ----------
        buffer : bytes
            The buffer storing the whole compressed `.nnf` file content.

        Returns
        -------
        string
            The string storing the whole decompressed `.nnf` file content.
        """
        # decompress nnf file
        while buffer[0] != b'='[0]:
            buffer = buffer[1:]
        buffer = buffer[2:]
        return bz2.decompress(buffer)[:-1].decode('ascii').strip()

    def _parse(self, nnf_file):
        """Parse the `.nnf` file
        
        Parameters
        ----------
        nnf_file : string
            The string storing the while decompressed `.nnf` file content.

        Returns
        -------
        list of dict
            Parsed setups as list of dictionary storing the parsed `.nnf` file content.
            Each dictionary in the list is the hyperparameters for a layer.
        """
        # parse input file
        parser = lark.Lark(r'''
        identifier : CNAME

        inputsize : "inputsize" "=" INT ";"

        assign : identifier "=" value ";"

        layer : "layer" "[" assign * "]"

        atom_net : "atom_net" WORD "$" layer * "$"

        start: inputsize atom_net

        value : INT
              | FLOAT
              | "FILE" ":" FILENAME "[" INT "]"

        FILENAME : ("_"|"-"|"."|LETTER|DIGIT)+

        %import common.SIGNED_NUMBER
        %import common.LETTER
        %import common.WORD
        %import common.DIGIT
        %import common.INT
        %import common.FLOAT
        %import common.CNAME
        %import common.WS
        %ignore WS
        ''')
        tree = parser.parse(nnf_file)

        # execute parse tree
        class TreeExec(lark.Transformer):

            def identifier(self, v):
                v = v[0].value
                return v

            def value(self, v):
                if len(v) == 1:
                    v = v[0]
                    if v.type == 'FILENAME':
                        v = v.value
                    elif v.type == 'INT':
                        v = int(v.value)
                    elif v.type == 'FLOAT':
                        v = float(v.value)
                    else:
                        raise ValueError('unexpected type')
                elif len(v) == 2:
                    v = self.value([v[0]]), self.value([v[1]])
                else:
                    raise ValueError('length of value can only be 1 or 2')
                return v

            def assign(self, v):
                name = v[0]
                value = v[1]
                return name, value

            def layer(self, v):
                return dict(v)

            def atom_net(self, v):
                layers = v[1:]
                return layers

            def start(self, v):
                return v[1]

        layer_setups = TreeExec().transform(tree)
        return layer_setups

    def _construct(self, setups, dirname):
        """Construct model from parsed setups
        
        Parameters
        ----------
        setups : list of dict
            Parsed setups as list of dictionary storing the parsed `.nnf` file content.
            Each dictionary in the list is the hyperparameters for a layer.
        dirname : string
            The directory where network files are stored.
        """

        # Activation defined in:
        # https://github.com/Jussmith01/NeuroChem/blob/master/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L868
        self.activation_index = None
        self.activation = None
        self.layers = len(setups)
        for i in range(self.layers):
            s = setups[i]
            in_size = s['blocksize']
            out_size = s['nodes']
            activation = s['activation']
            wfn, wsz = s['weights']
            bfn, bsz = s['biases']
            if i == self.layers-1:
                if activation != 6:  # no activation
                    raise ValueError('activation in the last layer must be 6')
            else:
                if self.activation_index is None:
                    if activation != 5:
                        raise NotImplementedError('only gaussian is supported')
                    else:
                        self.activation_index = activation
                        self.activation = lambda x: torch.exp(-x**2)
                elif self.activation_index != activation:
                    raise NotImplementedError(
                        'different activation on different layers are not supported')
            linear = nn.Linear(in_size, out_size).type(self.dtype)
            name = 'layer{}'.format(i)
            setattr(self, name, linear)
            if in_size * out_size != wsz or out_size != bsz:
                raise ValueError('bad parameter shape')
            wfn = os.path.join(dirname, wfn)
            bfn = os.path.join(dirname, bfn)
            self._load_param_file(linear, in_size, out_size, wfn, bfn)

    def _load_param_file(self, linear, in_size, out_size, wfn, bfn):
        """Load `.wparam` and `.bparam` files"""
        wsize = in_size * out_size
        fw = open(wfn, 'rb')
        float_w = struct.unpack('{}f'.format(wsize), fw.read())
        linear.weight = torch.nn.parameter.Parameter(torch.FloatTensor(
            float_w).type(self.dtype).view(out_size, in_size))
        fw.close()
        fb = open(bfn, 'rb')
        float_b = struct.unpack('{}f'.format(out_size), fb.read())
        linear.bias = torch.nn.parameter.Parameter(torch.FloatTensor(
            float_b).type(self.dtype).view(out_size))
        fb.close()
    
    def get_activations(self, aev, layer):
        """Compute the activation of the specified layer.
        
        Parameters
        ----------
        aev : torch.Tensor
            The pytorch tensor of AEV as input to this model.
        layer : int
            The layer whose activation is desired. The index starts at zero, that is
            `layer=0` means the `activation(layer0(aev))` instead of `aev`. If the given
            layer is larger than the total number of layers, then the activation of the last
            layer will be returned.

        Returns
        -------
        torch.Tensor
            The pytorch tensor of activations of specified layer.
        """
        y = aev
        for j in range(self.layers):
            linear = getattr(self, 'layer{}'.format(j))
            y = linear(y)
            y = self.activation(y)
            if j == layer:
                break
        if layer >= self.layers-1:
            linear = getattr(self, 'layer{}'.format(self.layers-1))
            y = linear(y)
        return y

    def forward(self, aev):
        """Compute output from aev"""
        return self.get_activations(aev, math.inf)

class ModelOnAEV(nn.Module):
    """Subclass of `torch.nn.Module` for the [xyz]->[aev]->[per_atom_y]->y pipeline.

    Attributes
    ----------
    aev_computer : AEVComputer
        The AEV computer.
    aev_length : int
        Length of AEV.
    model_X : nn.Module
        Model for species X. There should be one such attribute for each supported species.
    reducer : function
        Function of (input, dim)->output that reduce the input tensor along the given dimension
        to get an output tensor. This function will be called with the per atom output tensor
        with internal shape as input, and desired reduction dimension as dim, and should reduce
        the input into the tensor containing desired output.
    """

    def __init__(self, aev_computer, **kwargs):
        """Initialize object from manual setup or from NeuroChem network directory.

        The caller must set either `from_pync` in order to load from NeuroChem network directory,
        or set `per_species` and `reducer`.

        Parameters
        ----------
        aev_computer : AEVComputer
            The AEV computer.

        Other Parameters
        ----------------
        from_pync : string
            Path to the NeuroChem network directory. If this parameter is set, then `per_species` and
            `reducer` should not be set.
        per_species : dict
            Dictionary with supported species as keys and objects of `torch.nn.Model` as values, storing
            the model for each supported species. These models will finally become `model_X` attributes.
        reducer : function
            The desired `reducer` attribute.

        Raises
        ------
        ValueError
            If `from_pync`, `per_species`, and `reducer` are not properly set.
        """

        super(ModelOnAEV, self).__init__()
        if not isinstance(aev_computer, AEVComputer):
            raise TypeError(
                "NeuralNetworkPotential: aev_computer must be a subclass of AEVComputer")
        self.aev_computer = aev_computer
        self.aev_length = aev_computer.radial_length() + aev_computer.angular_length()

        if 'from_pync' in kwargs and 'per_species' not in kwargs and 'reducer' not in kwargs:
            network_dir = kwargs['from_pync']
            self.reducer = torch.sum
            for i in self.aev_computer.species:
                filename = os.path.join(network_dir, 'ANN-{}.nnf'.format(i))
                model_X = PerSpeciesFromNeuroChem(self.aev_computer.dtype, filename)
                setattr(self, 'model_' + i, model_X)
        elif 'from_pync' not in kwargs and 'per_species' in kwargs and 'reducer' in kwargs:
            per_species = kwargs['per_species']
            for i in per_species:
                setattr(self, 'model_' + i, per_species[i])
            self.reducer = kwargs['reducer']
        else:
            raise ValueError(
                'bad arguments when initializing ModelOnAEV')

    def forward(self, coordinates, species):
        """Feed forward
        
        Parameters
        ----------
        coordinates : torch.Tensor
            The pytorch tensor of shape (conformations, atoms, 3) storing
            the coordinates of all atoms of all conformations.
        species : list of string
            List of string storing the species for each atom.

        Returns
        -------
        torch.Tensor
            Pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
        radial_aev, angular_aev = self.aev_computer(coordinates, species)
        fullaev = torch.cat([radial_aev, angular_aev], dim=2)
        atoms = len(species)
        per_atom_outputs = []
        for i in range(atoms):
            s = species[i]
            y = fullaev[:, i, :]
            model_X = getattr(self, 'model_' + s)
            y = model_X(y)
            per_atom_outputs.append(y)

        per_atom_outputs = torch.stack(per_atom_outputs)
        molecule_output = self.reducer(per_atom_outputs, dim=0)
        return molecule_output
