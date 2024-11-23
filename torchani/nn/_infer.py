import warnings
import os
import itertools
import typing as tp
from itertools import accumulate

import torch
from torch import Tensor

from torchani.csrc import MNP_IS_INSTALLED
from torchani.nn._core import AtomicContainer, AtomicNetwork, TightCELU


def jit_unused_if_no_mnp():
    def decorator(func):
        if MNP_IS_INSTALLED:
            return torch.jit.export(func)
        return torch.jit.unused(func)

    return decorator


def _check_openmp_threads() -> None:
    if "OMP_NUM_THREADS" not in os.environ:
        warnings.warn(
            "OMP_NUM_THREADS not set."
            " MNP works best if OMP_NUM_THREADS >= 2."
            " You can set this variable by running 'export OMP_NUM_THREADS=4')"
            " or 'export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK' if using slurm"
        )
        return

    num_threads = int(os.environ["OMP_NUM_THREADS"])
    if num_threads <= 0:
        raise RuntimeError(f"OMP_NUM_THREADS set to an incorrect value: {num_threads}")


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
    r"""The inference-optimized analogue of a `torchani.nn.Ensemble`

    Combines all networks of an ensemble that correspond to the same element into a
    single `BmmAtomicNetwork`.

    As an example, if an ensemble has 8 models, and each model has 1 H-network
    and 1 C-network, all 8 H-networks and all 8 C-networks are fused into two
    networks: one single H-BmmAtomicNework and one single C-BmmAtomicNetwork.

    The resulting networks perform the same calculations but faster, and using less CUDA
    kernel calls, since the conversion avoids iteration over the ensemble members in
    python.

    The `BmmAtomicNetwork` modules consist of sequences of `BmmLinear`, which perform
    batched matrix multiplication (BMM).
    """

    def __init__(self, ensemble: AtomicContainer):
        super().__init__()
        self._MNP_IS_INSTALLED = MNP_IS_INSTALLED
        self.total_members_num = 1  # Operates as a single ANINetworks
        self.active_members_idxs = [0]
        self.num_species = ensemble.num_species
        if not hasattr(ensemble, "members"):
            raise TypeError("BmmEnsemble can only take an Ensemble as an input")
        symbols = tuple(ensemble.member(0).atomics.keys())
        self.atomics = torch.nn.ModuleList(
            [
                BmmAtomicNetwork([m.atomics[s] for m in ensemble.members])
                for s in symbols
            ]
        )

        # bookkeeping for optimization
        self._last_species: Tensor = torch.empty(1)
        self._idx_list: tp.List[Tensor] = _make_idx_list(
            self._last_species, self.num_species, []
        )

    def forward(
        self,
        elem_idxs: Tensor,
        aevs: Tensor,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        assert elem_idxs.shape == aevs.shape[:-1]
        assert aevs.shape[0] == 1, "BmmEnsemble only supports single-conformer inputs"

        # Initialize each elem_idxs if it has not been init or the species has changed
        if self._MNP_IS_INSTALLED:
            same_elem_idxs = _is_same_tensor_optimized(self._last_species, elem_idxs)
        else:
            same_elem_idxs = _is_same_tensor(self._last_species, elem_idxs)

        if not same_elem_idxs:
            self._idx_list = _make_idx_list(elem_idxs, self.num_species, self._idx_list)
        self._last_species = elem_idxs

        aevs = aevs.flatten(0, 1)
        energies = aevs.new_zeros(aevs.shape[0])
        for i, bmm_atomic in enumerate(self.atomics):
            if self._idx_list[i].shape[0] > 0:
                if not torch.jit.is_scripting():
                    torch.cuda.nvtx.range_push(f"bmm-species-{i}")
                input_ = aevs.index_select(0, self._idx_list[i])
                energies[self._idx_list[i]] = bmm_atomic(input_).flatten()
                if not torch.jit.is_scripting():
                    torch.cuda.nvtx.range_pop()
        energies = energies.view_as(elem_idxs)
        if not atomic:
            energies = energies.sum(dim=-1)
        return energies


class BmmAtomicNetwork(torch.nn.Module):
    r"""The inference-optimized analogue of an `AtomicNetwork`

    `BmmAtomicNetwork` instances are "combined" networks for a single element. Each
    combined network holds all networks associated with all the members of an ensemble.
    They consist on a sequence of `BmmLinear` layers with interleaved activation
    functions (simple multi-layer perceptrons or MLPs).
    """

    def __init__(self, networks: tp.Sequence[AtomicNetwork]):
        super().__init__()
        self.has_biases = networks[0].has_biases
        self.activation = networks[0].activation

        num_layers = len(networks[0].layers)
        for network in networks[1:]:
            if network.has_biases != self.has_biases:
                raise ValueError("All networks must have the same bias flag")
            if len(network.layers) != num_layers:
                raise ValueError("All networks must have the same number of layers")
            if not isinstance(network.activation, type(self.activation)):
                raise ValueError("All networks must have the same activation function")

        # "layers" is now a sequence of BmmLinear
        #  Not sure why this fail
        self.layers = torch.nn.ModuleList(
            [BmmLinear([n.layers[j] for n in networks]) for j in range(num_layers)]  # type: ignore  # noqa
        )
        self.final_layer = BmmLinear([n.final_layer for n in networks])
        self._num_batched_networks = len(networks)

    def forward(self, features: Tensor) -> Tensor:
        features = features.expand(self._num_batched_networks, -1, -1)
        for layer in self.layers:
            features = self.activation(layer(features))
        return self.final_layer(features).mean(0)


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

        has_bias = linears[0].bias is not None
        for linear in linears:
            if (linear.bias is not None) != has_bias:
                raise ValueError("All Linear modules must have the same bias flag")

        # Concatenate biases
        if has_bias:
            bias = [layer.bias.view(1, 1, -1).clone().detach() for layer in linears]
            self.bias = torch.nn.Parameter(torch.cat(bias))
            self._beta = 1
        else:
            self.bias = torch.nn.Parameter(torch.empty(1).view(1, 1, 1))
            self._beta = 0

        self.batch_size = self.weight.shape[0]
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[2]

    def forward(self, input_: Tensor) -> Tensor:
        return torch.baddbmm(self.bias, input_, self.weight, beta=self._beta)

    def extra_repr(self):
        r""":meta private:"""
        return (
            f"# batch_size={self.batch_size}"
            f"# in_features={self.in_features}"
            f"# out_features={self.out_features}"
            f"# bias={self.bias is not None}"
        )


class MNPNetworks(AtomicContainer):
    def __init__(self, module: AtomicContainer, use_mnp: bool = False):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("MNPNetworks needs a CUDA device to use CUDA Streams")

        # MNP strategy in general is hard to maintain so emit warnings
        if use_mnp:
            warnings.warn(
                "MNPNetworks with MNP C++ extension is experimental."
                " It has a complex implementation, and may be removed in the future."
            )
        else:
            warnings.warn(
                "MNPNetworks with no MNP C++ extension is not optimized."
                " It is meant as a proof of concept and may be removed in the future."
            )

        self._MNP_IS_INSTALLED = MNP_IS_INSTALLED
        self.total_members_num = 1
        self.active_members_idxs = [0]
        self.num_species = module.num_species

        # Detect "ensemble" case via duck typing
        self._is_bmm = hasattr(module, "members")
        if self._is_bmm:
            symbols = tuple(module.member(0).atomics.keys())
            self.atomics = torch.nn.ModuleList(
                [
                    BmmAtomicNetwork([m.atomics[s] for m in module.members])
                    for s in symbols
                ]
            )
        else:
            self.atomics = torch.nn.ModuleList(list(module.atomics.values()))
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

        if self._use_mnp:
            if not self._MNP_IS_INSTALLED:
                raise RuntimeError("The MNP C++ extension is not installed")

            # Check that the OpenMP environment variable is correctly set
            _check_openmp_threads()

            # Copy params from networks (reshape & transpose if from BmmEnsemble)
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
        activation = self.atomics[0].activation
        if not isinstance(activation, TightCELU):
            raise ValueError(
                f"Unsupported activation {type(activation)},"
                " only torchani.atomics.TightCELU is supported"
            )
        num_layers: tp.List[int] = []  # len: num_species
        weights: tp.List[Tensor] = []  # len: sum(num_species * num_layers[j])
        biases: tp.List[Tensor] = []  # len: sum(num_species * num_layers[j])
        for atomic in self.atomics:
            if not isinstance(atomic.activation, type(activation)):
                raise ValueError("All atomic networks must have the same activation fn")
            num_layers.append(len(atomic.layers) + 1)
            if isinstance(atomic, BmmAtomicNetwork):
                for layer in itertools.chain(atomic.layers, [atomic.final_layer]):
                    weights.append(layer.weight.clone().detach())
                    biases.append(layer.bias.clone().detach())
            elif isinstance(atomic, AtomicNetwork):
                for layer in itertools.chain(atomic.layers, [atomic.final_layer]):
                    weights.append(layer.weight.clone().detach().transpose(0, 1))
                    biases.append(layer.bias.clone().detach().unsqueeze(0))
            else:
                raise ValueError(f"Unsupported atomic network {type(atomic)}")
        return weights, biases, num_layers

    def forward(
        self,
        elem_idxs: Tensor,
        aevs: Tensor,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        assert elem_idxs.shape == aevs.shape[:-1]
        assert aevs.shape[0] == 1, "MNPNetworks only supports single-conformer inputs"
        assert not atomic, "MNPNetworks doesn't support atomic energies"
        aevs = aevs.flatten(0, 1)

        if self._MNP_IS_INSTALLED:
            same_species = _is_same_tensor_optimized(self._last_species, elem_idxs)
        else:
            same_species = _is_same_tensor(self._last_species, elem_idxs)

        if not same_species:
            self._idx_list = _make_idx_list(elem_idxs, self.num_species, self._idx_list)
        self._last_species = elem_idxs

        # pyMNP
        if not self._use_mnp:
            if torch.jit.is_scripting():
                raise RuntimeError("JIT-MNPNetworks only supported with use_mnp=True")
            return PythonMNP.apply(
                aevs, self._idx_list, self.atomics, self._stream_list
            )
        # cppMNP
        return self._cpp_mnp(aevs)

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
            0.1,
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
