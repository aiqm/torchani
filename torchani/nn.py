from .aev_base import AEVComputer
import torch
import bz2
import os
import lark
import struct
import copy
import math
from . import buildin_network_dir, buildin_model_prefix
from .benchmarked import BenchmarkedModule

# For python 2 compatibility
if not hasattr(math, 'inf'):
    math.inf = float('inf')


class PerSpeciesFromNeuroChem(torch.jit.ScriptModule):
    """Subclass of `torch.nn.Module` for the per atom aev->y transformation, loaded from NeuroChem network dir.

    Attributes
    ----------
    dtype : torch.dtype
        Pytorch data type for tensors
    device : torch.Device
        The device where tensors should be.
    layers : int
        Number of layers.
    output_length : int
        The length of output vector
    layerN : torch.nn.Linear
        Linear model for each layer.
    activation : function
        Function for computing the activation for all layers but the last layer.
    activation_index : int
        The NeuroChem index for activation.
    """

    def __init__(self, dtype, device, filename):
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
        self.device = device
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
                    self.activation_index = activation
                    if activation == 5:  # Gaussian
                        self.activation = lambda x: torch.exp(-x*x)
                    elif activation == 9:  # CELU
                        alpha = 0.1
                        self.activation = lambda x: torch.where(
                            x > 0, x, alpha * (torch.exp(x/alpha)-1))
                    else:
                        raise NotImplementedError(
                            'Unexpected activation {}'.format(activation))
                elif self.activation_index != activation:
                    raise NotImplementedError(
                        'different activation on different layers are not supported')
            linear = torch.nn.Linear(in_size, out_size).type(self.dtype)
            name = 'layer{}'.format(i)
            setattr(self, name, linear)
            if in_size * out_size != wsz or out_size != bsz:
                raise ValueError('bad parameter shape')
            wfn = os.path.join(dirname, wfn)
            bfn = os.path.join(dirname, bfn)
            self.output_length = out_size
            self._load_param_file(linear, in_size, out_size, wfn, bfn)

    def _load_param_file(self, linear, in_size, out_size, wfn, bfn):
        """Load `.wparam` and `.bparam` files"""
        wsize = in_size * out_size
        fw = open(wfn, 'rb')
        w = struct.unpack('{}f'.format(wsize), fw.read())
        w = torch.tensor(w, dtype=self.dtype, device=self.device).view(
            out_size, in_size)
        linear.weight = torch.nn.parameter.Parameter(w, requires_grad=True)
        fw.close()
        fb = open(bfn, 'rb')
        b = struct.unpack('{}f'.format(out_size), fb.read())
        b = torch.tensor(b, dtype=self.dtype,
                         device=self.device).view(out_size)
        linear.bias = torch.nn.parameter.Parameter(b, requires_grad=True)
        fb.close()

    def get_activations(self, aev, layer):
        """Compute the activation of the specified layer.

        Parameters
        ----------
        aev : torch.Tensor
            The pytorch tensor of shape (conformations, aev_length) storing AEV as input to this model.
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
        for j in range(self.layers-1):
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
        """Compute output from aev

        Parameters
        ----------
        aev : torch.Tensor
            The pytorch tensor of shape (conformations, aev_length) storing AEV as input to this model.

        Returns
        -------
        torch.Tensor
            The pytorch tensor of shape (conformations, output_length) for output.
        """
        return self.get_activations(aev, math.inf)


class ModelOnAEV(BenchmarkedModule):
    """Subclass of `torch.nn.Module` for the [xyz]->[aev]->[per_atom_y]->y pipeline.

    Attributes
    ----------
    aev_computer : AEVComputer
        The AEV computer.
    output_length : int
        The length of output vector
    derivative : boolean
        Whether to support computing the derivative w.r.t coordinates, i.e. d(output)/dR
    derivative_graph : boolean
        Whether to generate a graph for the derivative. This would be required only if the
        derivative is included as part of the loss function.
    model_X : nn.Module
        Model for species X. There should be one such attribute for each supported species.
    reducer : function
        Function of (input, dim)->output that reduce the input tensor along the given dimension
        to get an output tensor. This function will be called with the per atom output tensor
        with internal shape as input, and desired reduction dimension as dim, and should reduce
        the input into the tensor containing desired output.
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            aev : time spent on computing AEV.
            nn : time spent on computing output from AEV.
            derivative : time spend on computing derivative w.r.t. coordinates after the outputs
                is given. This key is only available if derivative computation is turned on.
            forward : total time for the forward pass
    """

    def __init__(self, aev_computer, derivative=False, derivative_graph=False, benchmark=False, **kwargs):
        """Initialize object from manual setup or from NeuroChem network directory.

        The caller must set either `from_nc` in order to load from NeuroChem network directory,
        or set `per_species` and `reducer`.

        Parameters
        ----------
        aev_computer : AEVComputer
            The AEV computer.
        derivative : boolean
            Whether to support computing the derivative w.r.t coordinates, i.e. d(output)/dR
        derivative_graph : boolean
            Whether to generate a graph for the derivative. This would be required only if the
            derivative is included as part of the loss function. This argument must be set to
            False if `derivative` is set to False.
        benchmark : boolean
            Whether to enable benchmarking

        Other Parameters
        ----------------
        from_nc : string
            Path to the NeuroChem network directory. If this parameter is set, then `per_species` and
            `reducer` should not be set. If set to `None`, then the network ship with torchani will be
            used.
        ensemble : int
            Number of models in the model ensemble. If this is not set, then `from_nc` would refer to
            the directory storing the model. If set to a number, then `from_nc` would refer to the prefix
            of directories.
        per_species : dict
            Dictionary with supported species as keys and objects of `torch.nn.Model` as values, storing
            the model for each supported species. These models will finally become `model_X` attributes.
        reducer : function
            The desired `reducer` attribute.

        Raises
        ------
        ValueError
            If `from_nc`, `per_species`, and `reducer` are not properly set.
        """

        super(ModelOnAEV, self).__init__(benchmark)
        self.derivative = derivative
        self.output_length = None
        if not derivative and derivative_graph:
            raise ValueError(
                'ModelOnAEV: can not create graph for derivative if the computation of derivative is turned off')
        self.derivative_graph = derivative_graph

        if benchmark:
            self.compute_aev = self._enable_benchmark(self.compute_aev, 'aev')
            self.aev_to_output = self._enable_benchmark(
                self.aev_to_output, 'nn')
            if derivative:
                self.compute_derivative = self._enable_benchmark(
                    self.compute_derivative, 'derivative')
            self.forward = self._enable_benchmark(self.forward, 'forward')

        if not isinstance(aev_computer, AEVComputer):
            raise TypeError(
                "ModelOnAEV: aev_computer must be a subclass of AEVComputer")
        self.aev_computer = aev_computer

        if 'from_nc' in kwargs and 'per_species' not in kwargs and 'reducer' not in kwargs:
            if 'ensemble' not in kwargs:
                if kwargs['from_nc'] is None:
                    kwargs['from_nc'] = buildin_network_dir
                network_dirs = [kwargs['from_nc']]
                self.suffixes = ['']
            else:
                if kwargs['from_nc'] is None:
                    kwargs['from_nc'] = buildin_model_prefix
                network_prefix = kwargs['from_nc']
                network_dirs = []
                self.suffixes = []
                for i in range(kwargs['ensemble']):
                    suffix = '{}'.format(i)
                    network_dir = os.path.join(
                        network_prefix+suffix, 'networks')
                    network_dirs.append(network_dir)
                    self.suffixes.append(suffix)

            self.reducer = torch.sum
            for network_dir, suffix in zip(network_dirs, self.suffixes):
                for i in self.aev_computer.species:
                    filename = os.path.join(
                        network_dir, 'ANN-{}.nnf'.format(i))
                    model_X = PerSpeciesFromNeuroChem(
                        self.aev_computer.dtype, self.aev_computer.device, filename)
                    if self.output_length is None:
                        self.output_length = model_X.output_length
                    elif self.output_length != model_X.output_length:
                        raise ValueError(
                            'output length of each atomic neural network must match')
                    setattr(self, 'model_' + i + suffix, model_X)
        elif 'from_nc' not in kwargs and 'per_species' in kwargs and 'reducer' in kwargs:
            self.suffixes = ['']
            per_species = kwargs['per_species']
            for i in per_species:
                model_X = per_species[i]
                if not hasattr(model_X, 'output_length'):
                    raise ValueError(
                        'atomic neural network must explicitly specify output length')
                elif self.output_length is None:
                    self.output_length = model_X.output_length
                elif self.output_length != model_X.output_length:
                    raise ValueError(
                        'output length of each atomic neural network must match')
                setattr(self, 'model_' + i, model_X)
            self.reducer = kwargs['reducer']
        else:
            raise ValueError(
                'ModelOnAEV: bad arguments when initializing ModelOnAEV')

        if derivative and self.output_length != 1:
            raise ValueError(
                'derivative can only be computed for output length 1')

    def compute_aev(self, coordinates, species):
        """Compute full AEV

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
            Pytorch tensor of shape (conformations, atoms, aev_length) storing
            the computed AEVs.
        """
        radial_aev, angular_aev = self.aev_computer(coordinates, species)
        fullaev = torch.cat([radial_aev, angular_aev], dim=2)
        return fullaev

    def aev_to_output(self, aev, species):
        """Compute output from aev

        Parameters
        ----------
        aev : torch.Tensor
            Pytorch tensor of shape (conformations, atoms, aev_length) storing
            the computed AEVs.
        species : list of string
            List of string storing the species for each atom.

        Returns
        -------
        torch.Tensor
            Pytorch tensor of shape (conformations, output_length) for the
            output of each conformation.
        """
        conformations = aev.shape[0]
        atoms = len(species)
        rev_species = species[::-1]
        species_dedup = sorted(
            set(species), key=self.aev_computer.species.index)
        per_species_outputs = []
        for s in species_dedup:
            begin = species.index(s)
            end = atoms - rev_species.index(s)
            y = aev[:, begin:end, :].contiguous(
            ).view(-1, self.aev_computer.aev_length)

            def apply_model(suffix):
                model_X = getattr(self, 'model_' + s + suffix)
                return model_X(y)
            ys = [apply_model(suffix) for suffix in self.suffixes]
            y = sum(ys) / len(ys)
            y = y.view(conformations, -1, self.output_length)
            per_species_outputs.append(y)

        per_species_outputs = torch.cat(per_species_outputs, dim=1)
        molecule_output = self.reducer(per_species_outputs, dim=1)
        return molecule_output

    def compute_derivative(self, output, coordinates):
        """Compute the gradient d(output)/d(coordinates)"""
        # Since different conformations are independent, computing
        # the derivatives of all outputs w.r.t. its own coordinate is equivalent
        # to compute the derivative of the sum of all outputs w.r.t. all coordinates.
        return torch.autograd.grad(output.sum(), coordinates, create_graph=self.derivative_graph)[0]

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
        torch.Tensor or (torch.Tensor, torch.Tensor)
            If derivative is turned off, then this function will return a pytorch
            tensor of shape (conformations, output_length) for the output of each
            conformation.
            If derivative is turned on, then this function will return a pair of
            pytorch tensors where the first tensor is the output tensor as when the
            derivative is off, and the second tensor is a tensor of shape
            (conformation, atoms, 3) storing the d(output)/dR.
        """
        if not self.derivative:
            coordinates = coordinates.detach()
        else:
            coordinates = torch.tensor(coordinates, requires_grad=True)
        _coordinates, _species = self.aev_computer.sort_by_species(
            coordinates, species)
        aev = self.compute_aev(_coordinates, _species)
        output = self.aev_to_output(aev, _species)
        if not self.derivative:
            return output
        else:
            derivative = self.compute_derivative(output, coordinates)
            return output, derivative

    def export_onnx(self, dirname):
        """Export atomic networks into onnx format

        Parameters
        ----------
        dirname : string
            Name of the directory to store exported networks.
        """

        aev_length = self.aev_computer.aev_length
        dummy_aev = torch.zeros(1, aev_length)
        for s in self.aev_computer.species:
            nn_onnx = os.path.join(dirname, '{}.proto'.format(s))
            model_X = getattr(self, 'model_' + s)
            torch.onnx.export(model_X, dummy_aev, nn_onnx)
