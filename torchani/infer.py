import typing as tp

import torch
from torch import Tensor

from torchani.utils import check_openmp_threads
from torchani.tuples import SpeciesEnergies
from torchani.nn import Ensemble, ANIModel
from torchani.csrc import MNP_IS_INSTALLED


if MNP_IS_INSTALLED:
    # We need to import torchani.mnp to tell PyTorch to initialize torch.ops.mnp
    from . import mnp  # type: ignore # noqa: F401


def jit_unused_if_no_mnp():
    def decorator(func):
        if MNP_IS_INSTALLED:
            return torch.jit.export(func)
        return torch.jit.unused(func)

    return decorator


@jit_unused_if_no_mnp()
def _is_same_tensor_optimized(last: Tensor, current: Tensor) -> bool:
    if torch.jit.is_scripting():
        return torch.ops.mnp.is_same_tensor(last, current)
    return last.data_ptr() == current.data_ptr()


def _is_same_tensor(last: Tensor, current: Tensor) -> bool:
    # Fallback to this, it is slow but it is the only available path if MNP is
    # not installed, at least until TorchScript supports data_ptr() (which is
    # probably never)
    same_shape = last.shape == current.shape
    if not same_shape:
        return False
    return bool((last == current).all().item())


def _build_new_idx_list(
    species: Tensor,
    num_species: int,
    idx_list: tp.List[Tensor],
) -> tp.List[Tensor]:
    species_ = species.flatten()
    with torch.no_grad():
        idx_list = [torch.empty(0) for i in range(num_species)]
        for i in range(num_species):
            mask = species_ == i
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
        self._MNP_IS_INSTALLED = MNP_IS_INSTALLED
        self.num_networks = 1  # BmmEnsemble operates as a single ANIModel
        self.num_batched_networks = ensemble.num_networks
        self.num_species = ensemble.num_species

        self.atomic_networks = torch.nn.ModuleList(
            [
                BmmAtomicNetwork([animodel[symbol] for animodel in ensemble])
                for symbol in ensemble[0]
            ]
        )

        # bookkeeping for optimization
        self._last_species: Tensor = torch.empty(1)
        self._idx_list: tp.List[Tensor] = _build_new_idx_list(
            self._last_species,
            self.num_species,
            [],
        )

    def forward(  # type: ignore
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
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
        if self._MNP_IS_INSTALLED:
            same_species = _is_same_tensor_optimized(self._last_species, species)
        else:
            same_species = _is_same_tensor(self._last_species, species)

        if not same_species:
            self._idx_list = _build_new_idx_list(
                species,
                self.num_species,
                self._idx_list,
            )
        self._last_species = species

        aev = aev.flatten(0, 1)
        atomic_energies = torch.zeros(aev.shape[0], dtype=aev.dtype, device=aev.device)
        for i, net in enumerate(self.atomic_networks):
            if self._idx_list[i].shape[0] > 0:
                if self._MNP_IS_INSTALLED:
                    self._nvtx_push(i)
                input_ = aev.index_select(0, self._idx_list[i])
                atomic_energies[self._idx_list[i]] = net(input_).flatten()
                if self._MNP_IS_INSTALLED:
                    self._nvtx_pop()
        atomic_energies = atomic_energies.unsqueeze(0)
        return atomic_energies

    @jit_unused_if_no_mnp()
    def _nvtx_pop(self) -> None:
        torch.ops.mnp.nvtx_range_pop()

    @jit_unused_if_no_mnp()
    def _nvtx_push(self, i: int) -> None:
        torch.ops.mnp.nvtx_range_push(f"network_{i}")


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
        event_list: tp.List[tp.Optional[torch.cuda.Event]] = [
            torch.cuda.Event() for i in range(num_species)
        ]
        current_stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event()
        start_event.record(current_stream)

        input_list = [None] * num_species
        output_list = [None] * num_species
        for i, net in enumerate(atomic_networks):
            if idx_list[i].shape[0] > 0:
                torch.cuda.nvtx.mark(f"species = {i}")
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
        event_list: tp.List[tp.Optional[torch.cuda.Event]] = [
            torch.cuda.Event() for j, _ in enumerate(stream_list)
        ]

        for i, output in enumerate(output_list):
            if output is not None:
                torch.cuda.nvtx.mark(f"backward species = {i}")
                stream_list[i].wait_event(start_event)
                with torch.cuda.stream(stream_list[i]):
                    grad_tmp = torch.autograd.grad(
                        output, input_list[i], grad_o.flatten().expand_as(output)
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
        if not torch.cuda.is_available():
            raise RuntimeError(
                "InferModel requires a CUDA device since it relies on CUDA Streams"
            )
        import warnings

        # Infer model in general is hard to maintain so we emit some
        # deprecation messages. In the future we may not support it anymore
        if use_mnp:
            msg = (
                "InferModel with MNP C++ extension is experimental."
                " It has a complex implementation, and may be removed in the future."
            )
        else:
            msg = (
                "InferModel with no MNP C++ extension is not optimized."
                " It is meant as a proof of concept and may be removed in the future."
            )
        warnings.warn(msg, category=DeprecationWarning)

        self._MNP_IS_INSTALLED = MNP_IS_INSTALLED
        self.num_networks = 1  # For compatibility with ANIModel and Ensemble API
        self.num_species = module.num_species

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

        # bookkeeping for optimization
        self._last_species: Tensor = torch.empty(1)
        self._idx_list: tp.List[Tensor] = _build_new_idx_list(
            self._last_species,
            self.num_species,
            [],
        )

        self._stream_list = [torch.cuda.Stream() for i in range(self.num_species)]

        # Variables used only to pass tensors to torch.ops.mnp when use_mnp=True
        self._weight_list: tp.List[Tensor] = []
        self._bias_list: tp.List[Tensor] = []
        self._num_layers_list: tp.List[int] = []
        self._start_layers_list: tp.List[int] = []
        # Alpha can't be zero for CELU, this denotes 'not set'
        self._celu_alpha: float = 0.0

        if self._use_mnp:
            # Check that the OpenMP environment variable is correctly set
            check_openmp_threads(verbose=False)
            if not self._MNP_IS_INSTALLED:
                raise RuntimeError("The MNP C++ extension is not installed")
            self._init_mnp()

    @torch.jit.unused
    def _init_mnp(self) -> None:
        # Copy weights and biases (and transform them if copying from a BmmEnsemble)
        weight_list, bias_list = self._copy_weights_and_biases()

        self._num_layers_list = [len(weight) for weight in weight_list]
        self._start_layers_list = [0] * self.num_species
        for i in range(self.num_species - 1):
            self._start_layers_list[i + 1] = (
                self._start_layers_list[i] + self._num_layers_list[i]
            )

        # Flatten weight and bias list
        self._weight_list = [
            torch.nn.Parameter(item) for sublist in weight_list for item in sublist
        ]
        self._bias_list = [
            torch.nn.Parameter(item) for sublist in bias_list for item in sublist
        ]

    @torch.jit.unused
    def _copy_weights_and_biases(
        self,
    ) -> tp.Tuple[tp.List[tp.List[Tensor]], tp.List[tp.List[Tensor]]]:
        weight_list: tp.List[tp.List[Tensor]] = []  # shape: (num_species, num_layers)
        bias_list: tp.List[tp.List[Tensor]] = []
        for i, atomic_network in enumerate(self.atomic_networks):
            weights: tp.List[Tensor] = []
            biases: tp.List[Tensor] = []
            for layer in atomic_network:
                layer_type = type(layer)
                # Note that clone().detach() converts Parameter into Tensor
                if layer_type is torch.nn.Linear:
                    if self._is_bmm:
                        raise ValueError(
                            "torch.nn.Linear layers only supported for non-Bmm models"
                        )
                    weights.append(layer.weight.clone().detach().transpose(0, 1))
                    biases.append(layer.bias.clone().detach().unsqueeze(0))
                elif layer_type is BmmLinear:
                    if not self._is_bmm:
                        raise ValueError(
                            "BmmLinear layers only supported for Bmm models"
                        )
                    weights.append(layer.weight.clone().detach())
                    biases.append(layer.bias.clone().detach())
                elif layer_type is torch.nn.CELU:
                    if self._celu_alpha == 0.0:
                        if layer.alpha == 0.0:
                            raise ValueError("alpha can't be 0.0 for CELU")
                        self._celu_alpha = layer.alpha
                    elif self._celu_alpha != layer.alpha:
                        raise ValueError("All CELU layers should have the same alpha")
                else:
                    raise ValueError(
                        f"Unsupported layer type {layer_type}"
                        " Supported layers types are:\n"
                        "- torch.nn.Linear (for non-bmm models)\n"
                        "- torchani.infer.BmmLinear (for bmm models)\n"
                        "- torch.nn.CELU"
                    )
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

        if self._MNP_IS_INSTALLED:
            same_species = _is_same_tensor_optimized(self._last_species, species)
        else:
            same_species = _is_same_tensor(self._last_species, species)

        if not same_species:
            self._idx_list = _build_new_idx_list(
                species,
                self.num_species,
                self._idx_list,
            )
        self._last_species = species

        # pyMNP code path
        if not self._use_mnp:
            if not torch.jit.is_scripting():
                energies = MultiNetFunction.apply(
                    aev,
                    self._idx_list,
                    self.atomic_networks,
                    self._stream_list,
                )
                return SpeciesEnergies(species, energies)
            raise RuntimeError(
                "InferModel can only be JIT-compiled if init with use_mnp=True"
            )

        # cppMNP code path
        energies = self._multi_net_function_mnp(aev)
        return SpeciesEnergies(species, energies)

    @jit_unused_if_no_mnp()
    def _multi_net_function_mnp(self, aev: Tensor) -> Tensor:
        return torch.ops.mnp.run(
            aev,
            self.num_species,
            self._num_layers_list,
            self._start_layers_list,
            self._idx_list,
            self._weight_list,
            self._bias_list,
            self._stream_list,
            self._is_bmm,
            self._celu_alpha,
        )

    @torch.jit.export
    def _atomic_energies(self, species_aev: tp.Tuple[Tensor, Tensor]) -> Tensor:
        raise NotImplementedError("Not implemented for InferModel")
