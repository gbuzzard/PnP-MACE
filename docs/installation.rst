.. highlight:: shell

============
Installation
============


Stable release
--------------

The pnp_mace package requires python 3.6 or higher, and it's recommended that you
install in a virtual environment.  If those conditions are satisfied, you can
install pnp_mace by running this command in your activated virtual environment in your terminal:

.. code-block:: console

    pip install pnp_mace

Then move on to `Running the Demos`_

Detailed installation instructions
==================================

If you don't know which version of python you have or whether you have a virtual environment or if you have pip,
you can follow these instructions:

1. Create a directory, say :code:`pnp_mace_code` and change into this directory (or change to the :code:`PnP-MACE/demo` directory if you've downloaded from github).

2. In a terminal, enter :code:`python --version`.  If this returns 2.X.Y, then use

.. code-block:: console

    python3 -m venv new_venv --clear  --upgrade-deps
    source new_venv/bin/activate

Otherwise use

.. code-block:: console

    python -m venv new_venv --clear  --upgrade-deps
    source new_venv/bin/activate

After this step, :code:`python` will be mapped to a version of python and supporting code that are appropriate for the pnp_mace package. Even if your original python mapped to version 2, now when you enter :code:`python --version` you should get 3.X.Y.

3. In the same terminal window, enter :code:`pip install pnp_mace`

Now the pnp_mace package is available in python using :code:`import pnp_mace` or
:code:`import pnp_mace as pnpm`

When you are finished using pnp_mace, enter :code:`deactivate` to exit this virtual
environment.

The next time you want to use pnp_mace, change to the directory you created
above and enter :code:`source new_venv/bin/activate`.  You do not need to use pip a second time.

.. _`Running the Demos`:

Running the demos
=================

1. Follow the installation instructions above and activate the virtual environment.

2. If you already have the :code:`PnP-MACE/demo` directory from :code:`https://github.com/gbuzzard/PnP-MACE` then change into that directory (you can get this by downloading the `tarball`_ and uncompress).  Otherwise change into the directory you created during installation and download all the files at :code:`https://github.com/gbuzzard/PnP-MACE/tree/master/demo`.

3. In your terminal window, enter :code:`pip install -r requirements_demo.txt`

4. In your terminal window, enter :code:`python ct.py`.  You should get some text output regarding the operation of the demo, followed shortly by some images demonstrating the reconstruction.  Repeat for the remaining demos.

From sources
------------

The sources for PnP-MACE can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/gbuzzard/PnP-MACE

(see https://git-scm.com/downloads to install git) or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/gbuzzard/PnP-MACE/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/gbuzzard/PnP-MACE
.. _tarball: https://github.com/gbuzzard/PnP-MACE/tarball/master
