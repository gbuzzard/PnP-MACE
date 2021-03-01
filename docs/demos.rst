=====
Demos
=====

The demos folder in the github repo (https://github.com/gbuzzard/PnP-MACE) includes simple demos illustrating an elementary approach to superresolution and a CT example with high-dynamic-range images.

These demos are not meant to be state-of-the-art either in terms of reconstruction quality or run-time performance.  Rather, they are meant to be an invitation to explore these algorithms, to try alternative priors, to understand the effect of parameters on the reconstruction, and to adapt the code to other applications.

The demo folder under https://github.com/gbuzzard/PnP-MACE includes both .py files to be run with python and Jupyter notebooks that can be opened in Google colaboratory for experimentation online.

See the instructions under :ref:`Running the Demos` in the Installation instructions to run the demos.

Select the demo names below to see more detailed descriptions.

.. autosummary::
   :toctree: _autosummary
   :template: demo_template.rst
   :caption: Demo Reference
   :recursive:

   demo.ct_mace
   demo.superres_mace
   demo.superres_pnp
   demo
