Tips
====

- Coordinates are in Angstrom, energies are in Hartree
    - Therefore, forces are in Hartree/Angstrom
- Species are indexed by 0, 1, 2, 3, ....
    - You could use :class:`torchani.SpeciesConverter` to convert from periodic table element index to 0, 1, 2, 3, ...
    - Builtin models has an argument ``periodic_table_index`` in its constructor. You can set this to ``True`` when
      constructing models to switch to periodic table indexing.