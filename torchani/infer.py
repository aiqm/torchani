# type: ignore
# This file is too dynamic to type-check correctly
import typing as tp
import warnings
import importlib.metadata

import torch
from torch import Tensor

from torchani.utils import check_openmp_threads
from torchani.tuples import SpeciesEnergies


mnp_is_installed = 'torchani.mnp' in importlib.metadata.metadata(
    __package__.split('.')[0]).get_all('Provides')

if mnp_is_installed:
    # We need to import torchani.mnp to tell PyTorch to initialize torch.ops.mnp
    from . import mnp  # type: ignore # noqa: F401
else:
    warnings.warn("mnp not installed")


class BmmEnsemble(torch.nn.Module):
    """
    Fuse all same networks of an ensemble into BmmNetworks, for example 8 same H networks will be fused into 1 BmmNetwork.
    BmmNetwork is composed of BmmLinear layers, which will perform Batch Matmul (bmm) instead of normal matmul
    to reduce the number of kernel calls.
    """
    def __init__(self, models):
        super().__init__()
        self.num_network = len(models[0])
        self.num_models = len(models)
        self.use_num_models = self.num_models
        self.last_species = torch.empty(1)
        self.idx_list = [torch.empty(0) for i in range(self.num_network)]
        # assert all models have the same networks as model[0]
        # and each network should have same architecture

        # networks
        bmm_networks = []
        for net_key, network in models[0].items():
            bmm_networks.append(BmmNetwork([model[net_key] for model in models]))
        self.net_list = torch.nn.ModuleList(bmm_networks)

    def forward(self, species_aev: tp.Tuple[Tensor, Tensor],  # type: ignore
                cell: tp.Optional[Tensor] = None,
                pbc: tp.Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        num_mol = species.shape[0]
        assert num_mol == 1, "InferModel currently only support inference for single molecule"

        self._check_if_idxlist_needs_updates_jittable(species)
        idx_list = self.idx_list
        aev = aev.flatten(0, 1)
        energy_list = torch.zeros(self.num_network, dtype=aev.dtype, device=aev.device)
        for i, net in enumerate(self.net_list):
            if idx_list[i].shape[0] > 0:
                # torch.cuda.nvtx.mark(f'species = {i}')
                torch.ops.mnp.nvtx_range_push(f"network_{i}")
                input_ = aev.index_select(0, idx_list[i])
                output = net(input_).flatten()
                energy_list[i] = torch.sum(output)
                torch.ops.mnp.nvtx_range_pop()

        mol_energies = torch.sum(energy_list, 0, True)
        return SpeciesEnergies(species, mol_energies)

    @torch.jit.export
    def _atomic_energies(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        num_mol = species.shape[0]
        assert num_mol == 1, "InferModel currently only support inference for single molecule"

        self._check_if_idxlist_needs_updates_jittable(species)
        idx_list = self.idx_list
        aev = aev.flatten(0, 1)
        energy_list = torch.zeros(aev.shape[0], dtype=aev.dtype, device=aev.device)
        for i, net in enumerate(self.net_list):
            if idx_list[i].shape[0] > 0:
                # torch.cuda.nvtx.mark(f'species = {i}')
                torch.ops.mnp.nvtx_range_push(f"network_{i}")
                input_ = aev.index_select(0, idx_list[i])
                output = net(input_).flatten()
                energy_list[idx_list[i]] = output
                torch.ops.mnp.nvtx_range_pop()

        atomic_energies = energy_list.unsqueeze(0)
        return atomic_energies

    @torch.jit.export
    def set_species(self, species):
        species_ = species.flatten()
        with torch.no_grad():
            self.idx_list = [torch.empty(0) for i in range(self.num_network)]
            for i in range(self.num_network):
                mask = (species_ == i)
                midx = mask.nonzero().flatten()
                if midx.shape[0] > 0:
                    self.idx_list[i] = midx

    @torch.jit.export
    def _check_if_idxlist_needs_updates_jittable(self, species):
        # initialize each species index if it has not been initialized
        # or the species has changed
        if not torch.ops.mnp.is_same_tensor(self.last_species, species):
            self.set_species(species)
            self.last_species = species

    @torch.jit.export
    def select_models(self, use_num_models: tp.Optional[int] = None):
        if use_num_models is None:
            use_num_models = self.num_models
            return
        assert use_num_models <= self.num_models, f"use_num_models {use_num_models} cannot be larger than size {self.num_models}"
        for net in self.net_list:
            net.select_models(use_num_models)
        self.use_num_models = use_num_models


class BmmNetwork(torch.nn.Module):
    """
    Multiple BmmLinear layers with activation function
    """
    def __init__(self, networks):
        super().__init__()
        layers = []
        self.num_models = len(networks)
        self.use_num_models = self.num_models
        for layer_idx, layer in enumerate(networks[0]):
            if isinstance(layer, torch.nn.Linear):
                layers.append(BmmLinear([net[layer_idx] for net in networks]))
            else:
                assert isinstance(layer, torch.nn.CELU) or isinstance(layer, torch.nn.GELU), "Currently only support CELU/GELU as activation function"
                layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        self.is_bmmlinear_layer = [isinstance(layer, BmmLinear) for layer in self.layers]

    def forward(self, input_):
        input_ = input_.expand(self.use_num_models, -1, -1)
        for layer in self.layers:
            input_ = layer(input_)
        return input_.mean(0)

    @torch.jit.export
    def select_models(self, use_num_models: int):
        for i, layer in enumerate(self.layers):
            if self.is_bmmlinear_layer[i] and hasattr(layer, "select_models"):
                layer.select_models(use_num_models)
        self.use_num_models = use_num_models


class BmmLinear(torch.nn.Module):
    """
    Batch Linear layer fuses multiple Linear layers that have same architecture and same input.
    input : (b x n x m)
    weight: (b x m x p)
    bias  : (b x 1 x p)
    out   : (b x n x p)
    """
    def __init__(self, linear_layers):
        super().__init__()
        self.num_models = len(linear_layers)
        self.use_num_models = self.num_models
        # assert each layer has same architecture
        weights = [layer.weight.unsqueeze(0).clone().detach() for layer in linear_layers]
        self.weights = torch.nn.Parameter(torch.cat(weights).transpose(1, 2))
        if linear_layers[0].bias is not None:
            bias = [layer.bias.view(1, 1, -1).clone().detach() for layer in linear_layers]
            self.bias = torch.nn.Parameter(torch.cat(bias))
            self.beta = 1
        else:
            self.bias = torch.nn.Parameter(torch.empty(1).view(1, 1, 1))
            self.beta = 0

    def forward(self, input_):
        # TODO, slicing weight and bias at every step is slow and useless
        weights = self.weights[:self.use_num_models, :, :]
        if self.beta > 0:
            bias = self.bias[:self.use_num_models, :, :]
        else:
            bias = self.bias
        return torch.baddbmm(bias, input_, weights, beta=self.beta)

    def extra_repr(self):
        return f"batch={self.weights.shape[0]}, in_features={self.weights.shape[1]}, out_features={self.weights.shape[2]}, bias={self.bias is not None}"

    @torch.jit.export
    def select_models(self, use_num_models: int):
        self.use_num_models = use_num_models


# ###########################################################################################
# The code below implements the MNP (Multi Net Parallel) functionality, aimed at paralleling multiple networks across distinct CUDA streams.
# However, given the complexity of this algorithm and its lack of generalizability, it should be deprecated in the future.
# ###########################################################################################


class MultiNetFunction(torch.autograd.Function):
    """
    Run Multiple Networks (HCNO..) on different streams, this is python implementation of MNP (Multi Net Parallel) autograd function, which
    actually cannot parallel between different species networks because of loop performance of dynamic interpretation of python language.

    There is no multiprocessing used here, whereas cpp version is implemented with OpenMP.
    """
    @staticmethod
    def forward(ctx, aev, num_network, idx_list, net_list, stream_list):
        assert num_network == len(idx_list)
        assert num_network == len(net_list)
        assert num_network == len(stream_list)
        energy_list = torch.zeros(num_network, dtype=aev.dtype, device=aev.device)
        event_list = [torch.cuda.Event() for i in range(num_network)]
        current_stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(current_stream)

        input_list = [None] * num_network
        output_list = [None] * num_network
        for i, net in enumerate(net_list):
            if idx_list[i].shape[0] > 0:
                torch.cuda.nvtx.mark(f'species = {i}')
                stream_list[i].wait_event(start_event)
                with torch.cuda.stream(stream_list[i]):
                    input_ = aev.index_select(0, idx_list[i]).requires_grad_()
                    with torch.enable_grad():
                        output = net(input_).flatten()
                    input_list[i] = input_
                    output_list[i] = output
                    energy_list[i] = torch.sum(output)
                event_list[i].record(stream_list[i])
            else:
                event_list[i] = None

        # sync default stream with events on different streams
        for event in event_list:
            if event is not None:
                current_stream.wait_event(event)

        ctx.num_network = num_network
        ctx.energy_list = energy_list
        ctx.stream_list = stream_list
        ctx.output_list = output_list
        ctx.input_list = input_list
        ctx.idx_list = idx_list
        ctx.aev = aev
        output = torch.sum(energy_list, 0, True)
        return output

    @staticmethod
    def backward(ctx, grad_o):
        num_network = ctx.num_network
        stream_list = ctx.stream_list
        output_list = ctx.output_list
        input_list = ctx.input_list
        idx_list = ctx.idx_list
        aev = ctx.aev
        aev_grad = torch.zeros_like(aev)

        current_stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(current_stream)
        event_list = [torch.cuda.Event() for i in range(num_network)]

        for i, output in enumerate(output_list):
            if output is not None:
                torch.cuda.nvtx.mark(f'backward species = {i}')
                stream_list[i].wait_event(start_event)
                with torch.cuda.stream(stream_list[i]):
                    grad_tmp = torch.autograd.grad(output, input_list[i], grad_o.flatten().expand_as(output))[0]
                    aev_grad[idx_list[i]] = grad_tmp
                event_list[i].record(stream_list[i])
            else:
                event_list[i] = None

        # sync default stream with events on different streams
        for event in event_list:
            if event is not None:
                current_stream.wait_event(event)

        return aev_grad, None, None, None, None


class InferModelBase(torch.nn.Module):
    """
    Note when jit is True:
    It is user's responsibility to manually call set_species() function before change to a different molecule.

    TODO: set_species() could be ommited once jit support tensor.data_ptr()
    """
    def __init__(self, num_network):
        super().__init__()

        self.last_species = torch.empty(1)
        assert torch.cuda.is_available(), "Infer model needs cuda is available"

        self.num_network = num_network
        self.idx_list = [torch.empty(0) for i in range(self.num_network)]
        self.stream_list = [torch.cuda.Stream() for i in range(self.num_network)]

        # holders for jit when use_mnp == False
        self.weight_list_: tp.List[Tensor] = [torch.empty(0)]
        self.bias_list_: tp.List[Tensor] = [torch.empty(0)]
        self.celu_alpha: float = float('inf')
        self.num_layers_list: tp.List[int] = [0]
        self.start_layers_list: tp.List[int] = [0]

    def forward(self, species_aev: tp.Tuple[Tensor, Tensor],  # type: ignore
                cell: tp.Optional[Tensor] = None,
                pbc: tp.Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        num_mol = species.shape[0]
        assert num_mol == 1, "InferModel currently only support inference for single molecule"
        if torch.jit.is_scripting():  # if in compilation (script) mode
            if self.use_mnp:
                mol_energies = self._single_mol_energies_jittable((species, aev))
            else:
                raise RuntimeError("JIT Infer Model only support use_mnp=True")
        else:
            mol_energies = self._single_mol_energies((species, aev))
        return SpeciesEnergies(species, mol_energies)

    @torch.jit.export
    def _single_mol_energies_jittable(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        species, aev = species_aev
        aev = aev.flatten(0, 1)
        self._check_if_idxlist_needs_updates_jittable(species)
        output = torch.ops.mnp.run(aev, self.num_network, self.num_layers_list, self.start_layers_list, self.idx_list, self.weight_list_, self.bias_list_, self.stream_list, self.is_bmm, self.celu_alpha)
        return output

    @torch.jit.unused
    def _single_mol_energies(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        species, aev = species_aev
        aev = aev.flatten(0, 1)
        self._check_if_idxlist_needs_updates(species)

        # torch.cuda.nvtx.range_push('Network')
        if not self.use_mnp:
            output = MultiNetFunction.apply(aev, self.num_network, self.idx_list, self.net_list, self.stream_list)
        else:
            output = torch.ops.mnp.run(aev, self.num_network, self.num_layers_list, self.start_layers_list, self.idx_list, self.weight_list_, self.bias_list_, self.stream_list, self.is_bmm, self.celu_alpha)
        # torch.cuda.nvtx.range_pop()
        return output

    @torch.jit.unused
    def _check_if_idxlist_needs_updates(self, species):
        # initialize each species index if it has not been initialized
        # or the species has changed
        if self.last_species.data_ptr() != species.data_ptr():
            self.set_species(species)
            self.last_species = species

    @torch.jit.export
    def _check_if_idxlist_needs_updates_jittable(self, species):
        # initialize each species index if it has not been initialized
        # or the species has changed
        if not torch.ops.mnp.is_same_tensor(self.last_species, species):
            self.set_species(species)
            self.last_species = species

    @torch.jit.export
    def set_species(self, species):
        species_ = species.flatten()
        with torch.no_grad():
            self.idx_list = [torch.empty(0) for i in range(self.num_network)]
            for i in range(self.num_network):
                mask = (species_ == i)
                midx = mask.nonzero().flatten()
                if midx.shape[0] > 0:
                    self.idx_list[i] = midx

    @torch.jit.unused
    def init_mnp(self):
        assert mnp_is_installed, "MNP extension is not installed"
        self.weight_list = []  # shape: [num_networks, num_layers]
        self.bias_list = []
        self.celu_alpha = None

        # copy weights and bias, and transform them if is ensemble
        self.copy_weight_bias()

        self.num_layers_list = [len(weights) for weights in self.weight_list]
        self.start_layers_list = [0] * self.num_network
        for i in range(self.num_network - 1):
            self.start_layers_list[i + 1] = self.start_layers_list[i] + self.num_layers_list[i]

        # flatten weight and bias list
        self.weight_list = torch.nn.ParameterList([torch.nn.Parameter(item) for sublist in self.weight_list for item in sublist])
        self.bias_list = torch.nn.ParameterList([torch.nn.Parameter(item) for sublist in self.bias_list for item in sublist])
        self.dummy_parameter = torch.nn.Parameter(torch.tensor(1.0))

        self.num_weight_list = len(self.weight_list)
        self.num_bias_list = len(self.bias_list)
        # self.weight_list is ParameterList, which could not be interpreted as List<Tensor>
        self.weight_list_ = [w for w in self.weight_list]
        self.bias_list_ = [b for b in self.bias_list]

        # check OpenMP environment variable
        check_openmp_threads()

        self.use_mnp = True

    # used when self.weight_list migrate to another device
    @torch.jit.export
    def mnp_migrate_device(self):
        for i in range(self.num_weight_list):
            self.weight_list_[i] = self.weight_list_[i].to(self.dummy_parameter.device)
        for i in range(self.num_bias_list):
            self.bias_list_[i] = self.bias_list_[i].to(self.dummy_parameter.device)

    @torch.jit.unused
    def copy_weight_bias(self):
        raise NotImplementedError("NotImplemented for InferModelBase")


class ANIInferModel(InferModelBase):
    """
    InferModel for a single ANI model, instead of an ensemble.
    """
    def __init__(self, modules, use_mnp=True):
        num_network = len(modules)
        super().__init__(num_network)

        self.is_bmm = False
        self.net_list = [m for (key, m) in modules]

        # mnp
        self.use_mnp = False
        if use_mnp:
            self.init_mnp()

    @torch.jit.unused
    def copy_weight_bias(self):
        for i, net in enumerate(self.net_list):
            weights = []
            biases = []
            for layer in net:
                if isinstance(layer, torch.nn.Linear):
                    weights.append(layer.weight.clone().detach().transpose(0, 1))
                    biases.append(layer.bias.clone().detach().unsqueeze(0))
                else:
                    assert isinstance(layer, torch.nn.CELU), "Currently only support CELU as activation function"
                    if self.celu_alpha is None:
                        self.celu_alpha = layer.alpha
                    else:
                        assert self.celu_alpha == layer.alpha, "All CELU layer should have same alpha"
            self.weight_list.append(weights)
            self.bias_list.append(biases)


class BmmEnsembleMNP(InferModelBase):
    """
    Fuse all same networks of an ensemble into BmmNetworks, for example 8 same H networks will be fused into 1 BmmNetwork.
    BmmNetwork is composed of BmmLinear layers, which will perform Batch Matmul (bmm) instead of normal matmul
    to reduce the number of kernel calls.
    """
    def __init__(self, models, use_mnp=True):
        num_network = len(models[0])
        super().__init__(num_network)
        # assert all models have the same networks as model[0]
        # and each network should have same architecture

        self.is_bmm = True
        # networks
        bmm_networks = []
        for net_key, network in models[0].items():
            bmm_networks.append(BmmNetwork([model[net_key] for model in models]))
        self.net_list = torch.nn.ModuleList(bmm_networks)

        # mnp
        self.use_mnp = False
        if use_mnp:
            self.init_mnp()

    @torch.jit.unused
    def copy_weight_bias(self):
        for i, net in enumerate(self.net_list):
            weights = []
            biases = []
            for layer in net.layers:
                if isinstance(layer, BmmLinear):
                    weights.append(layer.weights.clone().detach())
                    biases.append(layer.bias.clone().detach())
                else:
                    assert isinstance(layer, torch.nn.CELU), "Currently only support CELU as activation function"
                    if self.celu_alpha is None:
                        self.celu_alpha = layer.alpha
                    else:
                        assert self.celu_alpha == layer.alpha, "All CELU layer should have same alpha"
            self.weight_list.append(weights)
            self.bias_list.append(biases)
