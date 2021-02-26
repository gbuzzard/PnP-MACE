API Documentation
=================

All of the classes and methods are available from the top level import of pnp_mace.  That is, if you include :code:`import pnp_mace as pnpm` in a file, then you can use pnpm to access the methods in :code:`pnp_mace.utils` (for instance :code:`pnp_mace.utils.load_img` can be accessed as :code:`pnpm.load_img`.)  Likewise for the methods and classes in other submodules.

.. autosummary::
   :toctree: _autosummary
   :template: package.rst
   :caption: API Reference
   :recursive:

   pnp_mace.pnpadmm
   pnp_mace.equilibriumproblem
   pnp_mace.agent
   pnp_mace.forwardagent
   pnp_mace.prioragent
   pnp_mace.utils
   pnp_mace
