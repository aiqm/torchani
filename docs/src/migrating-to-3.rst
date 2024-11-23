.. _torchani-migrating:

.. currentmodule:: torchani

Migrating to TorchANI 3
=======================

If you were using a previous version of TorchANI you may need to update your code to
work with TorchANI 3. We strive to keep backwards compatibility, but some minor breaking
changes were necessary in order to support improvements in the models, dataset
management, etc. Minor versions changes attempt to be fully backwards compatible, and
breaking changes are reserved for major releases.

Here we document the most important changes. In many cases code will run as is, but some
warnings are emitted if the old, legacy API is being used, so we also provide
recommendations to use the new functions when appropriate.

General usage of built-in ANI models
------------------------------------

If you were previously calling torchani models as:

.. code-block:: python
    
    import torchani

    species_indices = torch.tensor([[0, 1, 1, 0]])
    coords = torch.tensor(...)
    model = torchani.models.ANI1x()

    species_indices, energies = model((species_indices, coords))
    # or
    energies = model((species_indices, coords)).energies

you may prefer to do instead:

.. code-block:: python
    
    import torchani
    from torchani import single_point

    atomic_nums = torch.tensor([[1, 6, 6, 1]])
    coords = torch.tensor(...)
    model = torchani.models.ANI1x()

    result = single_point(model, atomic_nums, coords)
    energies = result["energies"]

Here "single-point" is typical chemistry jargon for "calculation on fixed molecule
coords". This was changed since it allows models to output more than a single scalar
value, which is necessary e.g. for models that output charges. Additionally, the new
allows for outputting forces and hessians without any familiarity with torch (no need to
do anything with `torch.Tensor.requires_grad`). Calling a model directly is still
possible.

To output other quantities of interest use:

.. code-block:: python
    
    result = single_point(model, atomic_nums, coords, forces=True, hessians=True)
    atomic_charges = result["atomic_charges"]  # Only for models that support this
    energies = result["energies"]
    forces = result["forces"]
    hessians = result["hessians"]

The :obj:`~torchani.aev.AEVComputer`, ``ANIModel``, ``Ensemble``, and :obj:`~torchani.nn.SpeciesConverter` classes
------------------------------------------------------------------------------------------------------------------

If you were previously using these classes as:

.. code-block:: python
    
    import torchani
    aevc = torchani.AEVComputer(...)
    animodel = torchani.ANIModel(...)
    ensemble = torchani.Ensemble(...)
    converter = torchani.SpeciesConverter(...)

    _, idxs = converter((species, coords), cell, pbc)
    _, aevs = aevc((idxs, coords), cell, pbc)
    _, energies = animodel((idxs, aevs), cell, pbc)
    _, energies = ensemble((idxs, aevs), cell, pbc)

you should now do this instead:

.. code-block:: python
    
    import torchani
    converter = torchani.nn.SpeciesConverter(...)
    aevc = torchani.AEVComputer(...)
    # Note that the following classes have different names
    ani_nets = torchani.ANINetworks(...)
    ensemble = torchani.Ensemble(...)

    idxs = converter(atomic_nums)
    aevs = aevc(idxs, coords, cell, pbc)
    energies = animodel(idxs, aevs)
    energies = ensemble(ixs, aevs)

Note that ``torchani.nn.ANIModel`` has been renamed to :obj:`~torchani.nn.ANINetworks`.
The old signature is still supported for now, but it will be removed in the future

Extra notes on the :obj:`~torchani.aev.AEVComputer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible now to separate the ``AEVComputer`` and
:obj:`~torchani.neighbors.Neighborlist` parts of a calculation like this:

.. code-block:: python
    
    import torchani
    neighborlist = torchani.neighbors.AllPairs()
    aevc = torchani.AEVComputer.like_2x()
    converter = torchani.SpeciesConverter(("H", "C", "N", "O"))

    idxs = converter(atomic_nums)
    cutoff = aevc.radial.cutoff
    neighbors = neighborlist(cutoff, idxs, coords, cell, pbc)
    aevc = aevc.compute_from_neighbors(idxs, neighbors)

This may be useful if you are want to use the computed neighborlist in more modules
afterwards (e.g. pair potentials, or other neural networks).

Additionally, :obj:`~torchani.aev.AEVComputer` is now initialized with different inputs.
If you prefer the old signature you can use the
:obj:`~torchani.aev.AEVComputer.from_constants` constructor instead.

Usage of ``torchani.data``
--------------------------

This module is deprecated, you can still access it under `torchani.legacy_data`, but
its use is discouraged, and moving forward it will not be maintained. Use
`torchani.datasets` instead (it is similar to ``torchvision.datasets`` which you may
be familiar with).

Usage of :obj:`~torchani.nn.Sequential`
---------------------------------------

The `torchani.nn.Sequential` class is still available, but *its use is highly
discouraged*.

If you were previously doing:

.. code-block:: python

    import torchani
    aev_computer = torchani.aev.AEVComputer(...)  # Lots of arguments
    neural_networks = torchani.nn.ANIModel(...)  # Lots of arguments
    energy_shifter = torchani.utils.EnergyShifter(...)  # More arguments
    model = torchani.nn.Sequential(aev_computer, neural_networks, energy_shifter)

You should probably stop. This approach is error prone and verbose, and has multiple
gotchas.

The simplest way of creating a model for training, with random initial weights, is using
the factory functions in `torchani.arch`, such as `torchani.arch.simple_ani`:

.. code-block:: python

    from torchani.arch import simple_ani

    # LoT is used for the ground state energies, returned model is ready to train
    # Consult the documentation for the relevant options
    model = simple_ani(("H", "C", "N", "O", "S"), lot="wb97x-631gd")

These functions are wrappers over `torchani.arch.Assembler`, which you can also use
to create your model. For example, to create a model just like `torchani.models.ANI2x`,
but with random initial weights and the ``cuAEV`` strategy for faster training, do:

.. code-block:: python

    import torchani
    asm = torchani.arch.Assembler()
    asm.set_symbols(("H", "C", "N", "O"))
    # You can also pass your custom angular or radial terms as arguments
    asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy="cuaev")
    # make_2x_network is a function that, given a symbol, builds an atomic network,
    # you can pass whatever other function you want here.
    asm.set_atomic_networks(torchani.nn.make_2x_network)
    asm.set_gsaes_as_self_energies("wb97x-631gd")  # Add ground state atomic energies
    model = asm.assemble()  # The returned model is ready to train

This takes care of all the gotchas of building a model (for instance, it ensures the
``AEVComputer`` is initialized with the the correct number of elements, that it matches
the initial size of the networks, and that the internal order of the element idxs is the
same for all modules). It is a pretty customizable procedure, and has good defaults. It
also avoids having to return irrelevant outputs and accept irrelevant inputs in your
modules.

If you want even *more* flexibility, we recommend you create your own `torch.nn.Module`,
which is way easier than it sounds. As an example:

.. code-block:: python

    import torchani
    from torch.nn import Module

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.converter = torchani.nn.SpeciesConverter(...)
            self.neighborlist = torchani.neighbors.AllPairs(...)
            self.aevc = torchani.aev.AEVComputer(...)
            self.nn = torchani.nn.ANINetworks(...)
            self.shifter = torchani.sae.SelfEnergy(...)

        def forward(self, atomic_nums, coords, cell, pbc):
            idxs = self.converter(atomic_nums)
            cutoff = self.aevc.radial.cutoff
            neighbors = self.neighborlist(cutoff, idxs, coords, cell, pbc)
            aevs = self.aevc.compute_from_neighbors(idxs, neighbors)
            return self.nn(idxs, aevs) + self.shifter(idxs)

    model = Model()
    energies = model(atomic_nums, coords, cell, pbc)  # forward is automatically called

This gives you the full flexibility of `torch`, at the cost of some complexity.
