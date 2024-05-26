import typing as tp
from itertools import accumulate

import torch
from torch import Tensor

from torchani.utils import check_openmp_threads
from torchani.tuples import SpeciesEnergies
from torchani.csrc import MNP_IS_INSTALLED
from torchani.atomics import AtomicContainer


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
    # Potentially, slower fallback if MNP is not installed (until JIT supports data_ptr)
    same_shape = last.shape == current.shape
    if not same_shape:
        return False
    return bool((last == current).all().item())


def _make_idx_list(
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


class BmmEnsemble(AtomicContainer):
    r"""
    The inference-optimized analogue of an Ensemble, functions just like a
    single ANIModel.

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

    def __init__(self, ensemble: AtomicContainer):
        super().__init__()
        self._MNP_IS_INSTALLED = MNP_IS_INSTALLED
        self.num_networks = 1  # Operates as a single ANIModel
        self.num_batched_networks = ensemble.num_networks
        self.num_species = ensemble.num_species
        if not hasattr(ensemble, "members"):
            raise TypeError("BmmEnsemble can only take an Ensemble as an input")
        self.atomic_networks = torch.nn.ModuleList(
            [
                BmmAtomicNetwork(
                    [animodel.atomics[symbol] for animodel in ensemble.members]
                )
                for symbol in ensemble.member(0).atomics
            ]
        )

        # bookkeeping for optimization
        self._last_species: Tensor = torch.empty(1)
        self._idx_list: tp.List[Tensor] = _make_idx_list(
            self._last_species, self.num_species, []
        )

    def forward(
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
            self._idx_list = _make_idx_list(species, self.num_species, self._idx_list)
        self._last_species = species

        aev = aev.flatten(0, 1)
        atomic_energies = torch.zeros(aev.shape[0], dtype=aev.dtype, device=aev.device)
        for i, net in enumerate(self.atomic_networks):
            if self._idx_list[i].shape[0] > 0:
                if not torch.jit.is_scripting():
                    torch.cuda.nvtx.range_push(f"bmm-species-{i}")
                input_ = aev.index_select(0, self._idx_list[i])
                atomic_energies[self._idx_list[i]] = net(input_).flatten()
                if not torch.jit.is_scripting():
                    torch.cuda.nvtx.range_pop()
        atomic_energies = atomic_energies.unsqueeze(0)
        return atomic_energies


class BmmAtomicNetwork(torch.nn.Module):
    r"""
    The inference-optimized analogue of an atomic networks.

    BmmAtomicNetwork instances are "combined" atomic networks for a single
    element, each of which holds all networks associated with all the members
    of an ensemble. They consist on a sequence of BmmLinear layers with
    interleaved activation functions.

    BmmAtomicNetworks are used by BmmEnsemble to operate like a single ANIModel
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
    Batched Linear layer that fuses multiple Linear layers that have same architecture
    If "e" is the number of fused layers (which usually corresponds to members
    in an ensamble), then we have:

    input:  (e x n x m)
    weight: (e x m x p)
    bias:   (e x 1 x p)
    output: (e x n x p)
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


class InferModel(AtomicContainer):
    _is_bmm: bool
    _use_mnp: bool

    def __init__(self, module: AtomicContainer, use_mnp: bool = False):
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
        self.num_networks = 1
        self.num_species = module.num_species

        # In this case it is an Ensemble (note that we duck type this)
        if hasattr(module, "members"):
            self.atomic_networks = torch.nn.ModuleList(
                [
                    BmmAtomicNetwork(
                        [animodel.atomics[symbol] for animodel in module.members]
                    )
                    for symbol in module.member(0).atomics
                ]
            )
        else:
            self.atomic_networks = torch.nn.ModuleList(list(module.atomics.values()))
        self._is_bmm = hasattr(module, "members")

        self._use_mnp = use_mnp

        # Bookkeeping for optimization
        self._last_species: Tensor = torch.empty(1)
        self._idx_list: tp.List[Tensor] = _make_idx_list(
            self._last_species, self.num_species, []
        )

        # Stream list, to use multiple cuda streams (doesn't really work correctly)
        self._stream_list = [torch.cuda.Stream() for i in range(self.num_species)]

        # Variables used to pass tensors to torch.ops.mnp when use_mnp=True
        self._weight_names: tp.List[str] = []
        self._bias_names: tp.List[str] = []
        # Num linears (or bmm-linears) in each atomic network
        self._num_layers: tp.List[int] = []
        # Idx in the flattened weight/bias lists where each atomic network starts
        self._first_layer_idxs: tp.List[int] = []
        # Alpha can't be zero for CELU, this denotes 'not set'
        self._celu_alpha: float = 0.0

        if self._use_mnp:
            # Check that the OpenMP environment variable is correctly set
            check_openmp_threads(verbose=False)
            if not self._MNP_IS_INSTALLED:
                raise RuntimeError("The MNP C++ extension is not installed")
            # Copy params from networks (reshape & trans if copying from BmmEnsemble)
            weights, biases, self._num_layers = self._copy_weights_and_biases()

            # Cumulative sum from zero, over all the layers, gives the starting idxs
            self._first_layer_idxs = list(accumulate(self._num_layers[:-1], initial=0))

            # Flatten weight/bias lists and and register as buffers
            # (registration is so they behave correctly with to(*) methods)
            for j, w in enumerate(weights):
                name = f"weigth-{j}"
                self.register_buffer(name, w)
                self._weight_names.append(name)

            for j, b in enumerate(biases):
                name = f"bias-{j}"
                self.register_buffer(name, b)
                self._bias_names.append(name)

    @torch.jit.unused
    def _copy_weights_and_biases(
        self,
    ) -> tp.Tuple[tp.List[Tensor], tp.List[Tensor], tp.List[int]]:
        num_layers: tp.List[int] = []  # len: num_species
        weights: tp.List[Tensor] = []  # len: sum(num_species * num_layers[j])
        biases: tp.List[Tensor] = []  # len: sum(num_species * num_layers[j])
        for atomic_network in self.atomic_networks:
            num_layers.append(0)
            for layer in atomic_network:
                layer_type = type(layer)
                # *.clone().detach() converts torch.nn.Parameter into Tensor
                if layer_type is torch.nn.Linear and not self._is_bmm:
                    weights.append(layer.weight.clone().detach().transpose(0, 1))
                    biases.append(layer.bias.clone().detach().unsqueeze(0))
                    num_layers[-1] += 1
                elif layer_type is BmmLinear and self._is_bmm:
                    weights.append(layer.weight.clone().detach())
                    biases.append(layer.bias.clone().detach())
                    num_layers[-1] += 1
                elif layer_type is torch.nn.CELU:
                    if self._celu_alpha == 0.0:
                        self._celu_alpha = layer.alpha
                    elif self._celu_alpha != layer.alpha:
                        raise ValueError("All CELU layers must have the same alpha")
                else:
                    raise ValueError(
                        f"Unsupported layer type {layer_type}"
                        " Supported layers types are:\n"
                        "- torch.nn.Linear (for non-bmm models)\n"
                        "- torchani.infer.BmmLinear (for bmm models)\n"
                        "- torch.nn.CELU"
                    )
        return weights, biases, num_layers

    def forward(
        self,
        species_aev: tp.Tuple[Tensor, Tensor],
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
            self._idx_list = _make_idx_list(species, self.num_species, self._idx_list)
        self._last_species = species

        # pyMNP code path
        if not self._use_mnp:
            if not torch.jit.is_scripting():
                energies = PythonMNP.apply(
                    aev,
                    self._idx_list,
                    self.atomic_networks,
                    self._stream_list,
                )
                return SpeciesEnergies(species, energies)
            raise RuntimeError("JIT-InferModel only supported with use_mnp=True")

        # cppMNP code path
        energies = self._cpp_mnp(aev)
        return SpeciesEnergies(species, energies)

    @jit_unused_if_no_mnp()
    def _cpp_mnp(self, aev: Tensor) -> Tensor:
        weights: tp.List[Tensor] = []
        biases: tp.List[Tensor] = []
        for name, buffer in self.named_buffers():
            if name in self._bias_names:
                biases.append(buffer)
            elif name in self._weight_names:
                weights.append(buffer)
        return torch.ops.mnp.run(
            aev,
            self.num_species,
            self._num_layers,
            self._first_layer_idxs,
            self._idx_list,
            weights,
            biases,
            self._stream_list,
            self._is_bmm,
            self._celu_alpha,
        )


# ######################################################################################
# This code is a python implementation of the MNP (Multi Net Parallel) functionality,
# which is supposed to parallelize multiple networks across distinct CUDA streams.
# The python implementation is meant as a proof of concept and is not performant,
# in fact it is slower than a straightforward loop.
# This implementation has no multiprocessing, but OpenMP is used for the C++
# Implementation in torchani.csrc
# ######################################################################################
class PythonMNP(torch.autograd.Function):
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
