import math
import typing as tp
import warnings
import importlib.metadata

import torch
from torch import Tensor

from torchani.utils import check_openmp_threads
from torchani.tuples import SpeciesEnergies
from torchani.nn import Ensemble, ANIModel


mnp_is_installed = 'torchani.mnp' in importlib.metadata.metadata(
    __package__.split('.')[0]).get_all('Provides', [])

if mnp_is_installed:
    # We need to import torchani.mnp to tell PyTorch to initialize torch.ops.mnp
    from . import mnp  # type: ignore # noqa: F401
else:
    warnings.warn("mnp not installed")


def _is_same_tensor(last: Tensor, current: Tensor) -> bool:
    if torch.jit.is_scripting():
        return torch.ops.mnp.is_same_tensor(last, current)
    return last.data_ptr() == current.data_ptr()


def _build_new_idx_list(species: Tensor, num_species: int) -> tp.List[Tensor]:
    species_ = species.flatten()
    with torch.no_grad():
        idx_list = [torch.empty(0) for i in range(num_species)]
        for i in range(num_species):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                idx_list[i] = midx
    return idx_list


class BmmEnsemble(torch.nn.Module):
    r"""
    The inference-optimized analogue of an ANIModel.

    This class fuses all networks of an ensemble that correspond to the same
    element into one single BmmAtomicNetwork.

    As an example, if an ensemble has 8 models, and each model has 1 H-network
    and 1 C-network, all 8 H-networks and all 8 C-networks are fused into two
    networks: one single H-BmmAtomicNework and one single C-BmmAtomicNetwork.

    The resulting networks perform the same calculations but with less CUDA
    kernel calls, since iteration over the ensemble models in python is not
    needed, so this avoids the interpreter overhead.

    The BmmAtomicNetwork modules consist of sequences of BmmLinear, which
    perform Batched Matrix Multiplication (BMM).
    """
    num_networks: int
    num_species: int

    def __init__(self, ensemble: Ensemble):
        super().__init__()
        self.num_networks = 1  # BmmEnsemble operates as a single ANIModel
        self.num_batched_networks = ensemble.num_networks
        self.num_species = ensemble.num_species

        self.atomic_networks = torch.nn.ModuleList(
            [
                BmmAtomicNetwork([animodel[symbol] for animodel in ensemble])
                for symbol in ensemble[0]
            ]
        )

        # bookkeeping for optimization purposes
        self.last_species: Tensor = torch.empty(1)
        self.idx_list: tp.List[Tensor] = [
            torch.empty(0) for i in range(self.num_species)
        ]

    def forward(  # type: ignore
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None
    ) -> SpeciesEnergies:
        atomic_energies = self._atomic_energies(species_aev).squeeze(0)
        return SpeciesEnergies(species_aev[0], torch.sum(atomic_energies, 0, True))

    @torch.jit.export
    def _atomic_energies(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        assert aev.shape[0] == 1, "BmmEnsemble only supports single-conformer inputs"

        # Initialize each species index if it has not been initialized or the
        # species has changed
        if not _is_same_tensor(self.last_species, species):
            self.idx_list = _build_new_idx_list(species, self.num_species)
        self.last_species = species

        aev = aev.flatten(0, 1)
        atomic_energies = torch.zeros(aev.shape[0], dtype=aev.dtype, device=aev.device)
        for i, net in enumerate(self.atomic_networks):
            if self.idx_list[i].shape[0] > 0:
                torch.ops.mnp.nvtx_range_push(f"network_{i}")
                input_ = aev.index_select(0, self.idx_list[i])
                atomic_energies[self.idx_list[i]] = net(input_).flatten()
                torch.ops.mnp.nvtx_range_pop()
        atomic_energies = atomic_energies.unsqueeze(0)
        return atomic_energies


class BmmAtomicNetwork(torch.nn.Module):
    r"""
    The inference-optimized analogue of atomic networks.

    BmmAtomicNetwork instances are "combined" atomic networks for a single
    element, each of which holds all networks associated with all the members
    of an ensemble. They consist on a sequence of BmmLinear layers with
    interleaved activation functions.

    BmmAtomicNetworks are used by BmmEnsemble to operate like a normal ANIModel
    and avoid iterating over the ensemble members.
    """
    def __init__(self, networks: tp.Sequence[torch.nn.Sequential]):
        super().__init__()
        self.num_batched_networks = len(networks)

        layers = []
        for layer_idx, layer in enumerate(networks[0]):
            if type(layer) is torch.nn.Linear:
                layers.append(BmmLinear([net[layer_idx] for net in networks]))
            elif type(layer) in (torch.nn.CELU, torch.nn.GELU):
                layers.append(layer)
            else:
                raise ValueError("Only GELU/CELU act. fn and Linear layers supported")

        # "layers" is now a sequence of BmmLinear interleaved with activation functions
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = input_.expand(self.num_batched_networks, -1, -1)
        for layer in self.layers:
            input_ = layer(input_)
        return input_.mean(0)

    def __iter__(self) -> tp.Iterator[torch.nn.Module]:
        return iter(self.layers)


class BmmLinear(torch.nn.Module):
    """
    Batch Linear layer fuses multiple Linear layers that have same architecture
    If "b" is the number of fused layers (which usually corresponds to members
    in an ensamble), then we have:

    input:  (b x n x m)
    weight: (b x m x p)
    bias:   (b x 1 x p)
    output: (b x n x p)
    """
    def __init__(self, linears: tp.Sequence[torch.nn.Linear]):
        super().__init__()
        # Concatenate weights
        weights = [layer.weight.unsqueeze(0).clone().detach() for layer in linears]
        self.weight = torch.nn.Parameter(torch.cat(weights).transpose(1, 2))

        # Concatenate biases
        if linears[0].bias is not None:
            bias = [layer.bias.view(1, 1, -1).clone().detach() for layer in linears]
            self.bias = torch.nn.Parameter(torch.cat(bias))
            self.beta = 1
        else:
            self.bias = torch.nn.Parameter(torch.empty(1).view(1, 1, 1))
            self.beta = 0

    def forward(self, input_: Tensor) -> Tensor:
        return torch.baddbmm(self.bias, input_, self.weight, beta=self.beta)

    def extra_repr(self):
        return (
            f"batch={self.weight.shape[0]},"
            f" in_features={self.weight.shape[1]},"
            f" out_features={self.weight.shape[2]},"
            f" bias={self.bias is not None}"
        )


# ######################################################################################
# The code below implements the MNP (Multi Net Parallel) functionality, aimed
# at paralleling multiple networks across distinct CUDA streams. However, given
# the complexity of this algorithm and its lack of generalizability, it is
# deprecated and will be removed in the future
# ######################################################################################
class MultiNetFunction(torch.autograd.Function):
    r"""
    Run Multiple Networks (HCNO..) on different streams, this is python
    implementation of MNP (Multi Net Parallel) autograd function, which
    actually cannot parallel between different species networks because of loop
    performance of dynamic interpretation of python language.

    There is no multiprocessing used here, whereas cpp version is implemented
    with OpenMP.
    """
    @staticmethod
    def forward(
        ctx,
        aev,
        idx_list,
        atomic_networks,
        stream_list,
    ):
        num_species = len(atomic_networks)
        assert num_species == len(atomic_networks)
        assert num_species == len(stream_list)
        energy_list = torch.zeros(num_species, dtype=aev.dtype, device=aev.device)
        event_list: tp.List[tp.Optional[torch.cuda.Event]] = [torch.cuda.Event() for i in range(num_species)]
        current_stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(current_stream)

        input_list = [None] * num_species
        output_list = [None] * num_species
        for i, net in enumerate(atomic_networks):
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
                event = event_list[i]
                assert event is not None  # mypy
                event.record(stream_list[i])
            else:
                event_list[i] = None

        # sync default stream with events on different streams
        for event in event_list:
            if event is not None:
                current_stream.wait_event(event)

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
        stream_list = ctx.stream_list
        output_list = ctx.output_list
        input_list = ctx.input_list
        idx_list = ctx.idx_list
        aev = ctx.aev
        aev_grad = torch.zeros_like(aev)

        current_stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(current_stream)
        event_list: tp.List[tp.Optional[torch.cuda.Event]] = [torch.cuda.Event() for j, _ in enumerate(stream_list)]

        for i, output in enumerate(output_list):
            if output is not None:
                torch.cuda.nvtx.mark(f'backward species = {i}')
                stream_list[i].wait_event(start_event)
                with torch.cuda.stream(stream_list[i]):
                    grad_tmp = torch.autograd.grad(
                        output,
                        input_list[i],
                        grad_o.flatten().expand_as(output)
                    )[0]
                    aev_grad[idx_list[i]] = grad_tmp
                event = event_list[i]
                assert event is not None  # mypy
                event.record(stream_list[i])
            else:
                event_list[i] = None

        # sync default stream with events on different streams
        for event in event_list:
            if event is not None:
                current_stream.wait_event(event)
        return aev_grad, None, None, None


class InferModel(torch.nn.Module):
    num_networks: int
    num_species: int
    _is_bmm: bool
    _use_mnp: bool

    def __init__(self, module: tp.Union[Ensemble, ANIModel], use_mnp: bool = False):
        super().__init__()
        self.num_networks = 1  # For compatibility with ANIModel and Ensemble API
        self.num_species = module.num_species

        if not torch.cuda.is_available():
            raise RuntimeError("InferModel requires a CUDA device")

        if isinstance(module, Ensemble):
            self.atomic_networks = torch.nn.ModuleList(
                [
                    BmmAtomicNetwork([animodel[symbol] for animodel in module])
                    for symbol in module[0]
                ]
            )
        else:
            self.atomic_networks = torch.nn.ModuleList(list(module.values()))
        self._is_bmm = isinstance(module, Ensemble)
        self._use_mnp = use_mnp

        # bookkeeping for optimization purposes
        self.last_species = torch.empty(1)
        self.idx_list = [torch.empty(0) for i in range(self.num_species)]

        self.stream_list = [torch.cuda.Stream() for i in range(self.num_species)]

        # Holders for jit when use_mnp == False
        self.weight_list: tp.List[Tensor] = [torch.empty(0)]
        self.bias_list: tp.List[Tensor] = [torch.empty(0)]
        self.celu_alpha: float = float('inf')
        self.num_layers_list: tp.List[int] = [0]
        self.start_layers_list: tp.List[int] = [0]

        if self._use_mnp:
            self._init_mnp()

    @torch.jit.unused
    def _init_mnp(self) -> None:
        if not mnp_is_installed:
            raise RuntimeError("MNP extension is not installed")
        # Copy weights and biases (and transform them if copying from a BmmEnsemble)
        weight_list, bias_list = self._copy_weights_and_biases()

        self.num_layers_list = [len(weight) for weight in weight_list]
        self.start_layers_list = [0] * self.num_species
        for i in range(self.num_species - 1):
            self.start_layers_list[i + 1] = self.start_layers_list[i] + self.num_layers_list[i]

        # Flatten weight and bias list
        self.weight_list = [torch.nn.Parameter(item) for sublist in weight_list for item in sublist]
        self.bias_list = [torch.nn.Parameter(item) for sublist in bias_list for item in sublist]

        # Check that the OpenMP environment variable is correctly set
        check_openmp_threads(verbose=False)

    @torch.jit.unused
    def _copy_weights_and_biases(self) -> tp.Tuple[tp.List[tp.List[Tensor]], tp.List[tp.List[Tensor]]]:
        weight_list: tp.List[tp.List[Tensor]] = []  # shape: (num_species, num_layers)
        bias_list: tp.List[tp.List[Tensor]] = []
        for i, atomic_network in enumerate(self.atomic_networks):
            weights: tp.List[Tensor] = []
            biases: tp.List[Tensor] = []
            for layer in atomic_network:
                layer_type = type(layer)
                # Note that clone().detach() converts Parameter into Tensor
                if layer_type is torch.nn.Linear:
                    weights.append(layer.weight.clone().detach().transpose(0, 1))
                    biases.append(layer.bias.clone().detach().unsqueeze(0))
                elif layer_type is BmmLinear:
                    weights.append(layer.weight.clone().detach())
                    biases.append(layer.bias.clone().detach())
                elif layer_type is torch.nn.CELU:
                    if math.isinf(self.celu_alpha):
                        self.celu_alpha = layer.alpha
                    else:
                        if self.celu_alpha != layer.alpha:
                            raise ValueError("All CELU layers should have the same alpha")
                else:
                    raise ValueError(f"Unsupported layer type {layer_type}, only supported layers are Linear, BmmLinear and CELU")
            weight_list.append(weights)
            bias_list.append(biases)
        return weight_list, bias_list

    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],  # type: ignore
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        assert aev.shape[0] == 1, "InferModel only supports single-conformer inputs"
        aev = aev.flatten(0, 1)

        if not _is_same_tensor(self.last_species, species):
            self.idx_list = _build_new_idx_list(species, self.num_species)
        self.last_species = species

        if self._use_mnp:
            energies = torch.ops.mnp.run(
                aev,
                self.num_species,
                self.num_layers_list,
                self.start_layers_list,
                self.idx_list,
                self.weight_list,
                self.bias_list,
                self.stream_list,
                self._is_bmm,
                self.celu_alpha,
            )
        else:
            if torch.jit.is_scripting():
                raise RuntimeError("JIT Infer Model only supports use_mnp=True")
            else:
                energies = MultiNetFunction.apply(
                    aev,
                    self.idx_list,
                    self.atomic_networks,
                    self.stream_list,
                )
        return SpeciesEnergies(species, energies)

    @torch.jit.export
    def _atomic_energies(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        raise NotImplementedError("Not implemented for InferModel")
