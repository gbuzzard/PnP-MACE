
.. image:: https://raw.githubusercontent.com/gbuzzard/PnP-MACE/master/docs/_static/PnP-MACE-Logo.svg
   :width: 320px
   :alt: PnP-MACE logo
   :align: center


..
    .. image:: https://img.shields.io/pypi/v/pnp_mace.svg
        :target: https://pypi.python.org/pypi/pnp_mace


.. image:: https://travis-ci.com/gbuzzard/PnP-MACE.svg?branch=master
    :target: https://travis-ci.com/gbuzzard/PnP-MACE

.. image:: https://readthedocs.org/projects/pnp-mace/badge/?version=latest
    :target: https://pnp-mace.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

|


.. docs-include-ref

Introduction
------------

This python package provides methods and utilities to explore the PnP algorithm and MACE framework in the context
of image reconstruction problems, along with some simple demos.  The ideas leading to this package are outlined in

* `A SIAM News overview article from March 2021`_
* `The 2020 SIAM Imaging Sciences Best Paper Award lecture`_

.. _`A SIAM News overview article from March 2021`: https://sinews.siam.org/Details-Page/plug-and-play-a-general-approach-for-the-fusion-of-sensor-and-machine-learning-models
.. _`The 2020 SIAM Imaging Sciences Best Paper Award lecture`: https://www.youtube.com/watch?v=GjCmxTqAJDo&feature=youtu.be

with more detail in `the papers in the references`_.

.. _`the papers in the references`: https://pnp-mace.readthedocs.io/en/latest/zreferences.html

Documentation on this package is available at https://pnp-mace.readthedocs.io, which
includes `installation instructions`_.

Demo files are available as python scripts and Jupyter notebooks.  See the `Demos page`_ for more details.

.. _`Demos page`: https://pnp-mace.readthedocs.io/en/latest/demos.html

.. _`installation instructions`: https://pnp-mace.readthedocs.io/en/latest/installation.html


Features
========

* Implementation of basic Plug-and-Play method as described in

    `Plug-and-Play Priors for Bright Field Electron Tomography and Sparse Interpolation,
    *IEEE Transactions on Computational Imaging*, vol. 2, no. 4, Dec. 2016.`__

__  https://engineering.purdue.edu/~bouman/publications/orig-pdf/tci05.pdf

* Implementation of the solution of MACE equations using Mann iterations as in

    `Plug-and-Play Unplugged: Optimization Free Reconstruction using Consensus Equilibrium,
    *SIAM Journal on Imaging Sciences*, vol. 11, no. 3, pp. 2001-2020, Sept. 2018.`__

__ https://engineering.purdue.edu/~bouman/publications/orig-pdf/SIIMS01.pdf

* `Demo code illustrating reconstruction with image subsampling and tomography`_.

.. _`Demo code illustrating reconstruction with image subsampling and tomography`: https://pnp-mace.readthedocs.io/en/latest/demos.html

This is free software with a `BSD license`_.

.. _`BSD license`: https://github.com/gbuzzard/PnP-MACE/blob/master/LICENSE
