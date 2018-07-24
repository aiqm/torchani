from .. import _six
import os
import bz2
import lark
import torch
import math

class NeuroChemAtomicNetwork(torch.jit.ScriptModule):
    """Per atom aev->y transformation, loaded from NeuroChem network dir.

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
        Function for computing the activation for all layers but the
        last layer.
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
            The file name for the `.nnf` file that store network
            hyperparameters. The `.bparam` and `.wparam` must be
            in the same directory
        """
        super(NeuroChemAtomicNetwork, self).__init__()

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
            Parsed setups as list of dictionary storing the parsed `.nnf`
            file content. Each dictionary in the list is the hyperparameters
            for a layer.
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
            Parsed setups as list of dictionary storing the parsed `.nnf`
            file content. Each dictionary in the list is the hyperparameters
            for a layer.
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
                        '''different activation on different
                        layers are not supported''')
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
            The pytorch tensor of shape (conformations, aev_length) storing AEV
            as input to this model.
        layer : int
            The layer whose activation is desired. The index starts at zero,
            that is `layer=0` means the `activation(layer0(aev))` instead of
            `aev`. If the given layer is larger than the total number of
            layers, then the activation of the last layer will be returned.

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
            The pytorch tensor of shape (conformations, aev_length) storing
            AEV as input to this model.

        Returns
        -------
        torch.Tensor
            The pytorch tensor of shape (conformations, output_length) for
            output.
        """
        return self.get_activations(aev, math.inf)
