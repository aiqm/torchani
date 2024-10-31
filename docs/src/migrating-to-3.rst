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

you should now do this instead:

.. code-block:: python
    
    import torchani
    atomic_nums = torch.tensor([[1, 6, 6, 1]])
    coords = torch.tensor(...)
    model = torchani.models.ANI1x()

    result = model.sp(atomic_nums, coords)
    energies = result["energies"]

Here "sp" stands for a "single-point calculation" (typical chemistry jargon). This was
changed since it allows models to output more than a single scalar value, which is
necessary e.g. for models that output charges. Additionally, the new version is simpler
and allows for outputting forces and hessians without any familiarity with torch (no
need to do anything with `torch.Tensor.requires_grad`). Calling a model directly is
still possible, but *discouraged*.

To output other quantities of interest use:

.. code-block:: python
    
    result = model.sp(atomic_nums, coords, forces=True, hessians=True)
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
    ensemble = torchani.ANIEnsemble(...)

    idxs = converter(atomic_nums)
    aevs = aevc(idxs, coords, cell, pbc)
    energies = animodel(idxs, aevs)
    energies = ensemble(ixs, aevs)

.. NOTE The old behavior is also supported directly by using the old class names but we
   omit to mention this here on purpose.

Note that now the class names are :obj:`~torchani.nn.ANINetworks` instead of
``ANIModel``, and :obj:`~torchani.nn.ANIEnsemble` instead of ``Ensemble``.

The old signature is still supported by using the ``.call()`` method, but this is
discouraged. An example:

.. code-block:: python

    aevc = torchani.AEVComputer(...)
    _, aevs = aevc.call((species, coords), cell, pbc)  # Possible, but not recommended

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
    neighbors = neighborlist(idxs, coords, cell, pbc)
    aevc = aevc.compute_from_neighbors(idxs, neighbors)

This may be useful if you are want to use the computed neighborlist in more modules
afterwards (e.g. pair potentials, or other neural networks).

Additionally, :obj:`~torchani.aev.AEVComputer` is now initialized with different inputs.
If you prefer the old signature you can use the
:obj:`~torchani.aev.AEVComputer.from_constants` constructor instead (we recommend using
the new constructors however).

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
    aev_computer = torchani.AEVComputer(...)  # Lots of arguments
    neural_networks = torchani.ANIModel(...)  # Lots of arguments
    energy_shifter = torchani.EnergyShifter(...)  # More arguments
    model = torchani.nn.Sequential(aev_computer, neural_networks, energy_shifter)

You should probably stop. This approach is error prone and verbose, and has multiple
gotchas.

The simplest way of creating a model for training, with random initial weights, is using
the factory functions in `torchani.assembly`, such as `torchani.assembly.simple_ani`:

.. code-block:: python

    from torchani.assembly import simple_ani

    # LoT is used for the ground state energies
    model = simple_ani(lot="wb97x-631gd", symbols=("H", "C", "N", "O", "S"))
    # The returned model is ready to train
    # Consult the documentation for the relevant options

These functions are wrappers over `torchani.assembly.Assembler`, which you can also use
to create your model. For example, to create a model just like `torchani.models.ANI2x`,
but with random initial weights and the ``cuAEV`` strategy for faster training, do:

.. code-block:: python

    from torchani import assembly
    from torchani.nn import make_2x_network

    asm = assembly.Assembler()
    asm.set_symbols(("H", "C", "N", "O"))
    # You can also pass your custom angular or radial terms as arguments
    asm.set_aev_computer(radial_terms="ani2x", angular_terms="ani2x", strategy="cuaev")
    # make_2x_network is a function that, given a symbol, builds an atomic network,
    # you can pass whatever other function you want here.
    asm.set_atomic_networks(make_2x_network)
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
            self.converter = torchani.nn.SpeciesConverter(...)
            self.neighborlist = torchani.neighbors.AllPairs(...)
            self.aevc = torchani.aev.AEVComputer(...)
            self.nn = torchani.nn.ANINetworks(...)
            self.adder = torchani.potentials.SelfEnergy(...)

        def forward(self, atomic_nums, coords, cell, pbc):
            idxs = self.converter(atomic_nums)
            neighbors = self.neighborlist(idxs, coords, cell, pbc)
            aevs = self.aevc(idxs, neighbors)
            energies = self.nn(idxs, aevs)
            energies += self.adder(idxs)
            return energies

    model = Model()
    energies = model(atomic_nums, coords, cell, pbc)  # forward is automatically called

This gives you the full flexibility of `torch`, at the cost of some complexity.
