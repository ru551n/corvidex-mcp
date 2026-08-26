Repository layout
=================

The repository is organised into one directory per concern. Hardware
description sources live in ``rtl/``; each design is a single VHDL
file named after its top-level entity. Documentation lives in ``docs/``
as reStructuredText. Build and simulation scripts live in ``scripts/``.

Naming conventions
------------------

Module file names use lowercase letters and underscores, and must match
the top-level entity name. Signals use lowercase with underscores;
generic parameters use uppercase. Every public entity gets a short
header comment describing its function.

Adding a new module
-------------------

Create ``rtl/<name>.vhd`` with the entity and architecture, add a
section in this document if the module is user-facing, and add a
simulation entry in ``scripts/sim_setup.py``.