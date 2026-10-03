Installation
============

ANTsNormalizingFlows requires Python 3.10 or newer, NumPy, and PyTorch.
The Python import name is ``antsnormflows``. A GPU is optional; the getting
started example runs on CPU and does not require external data.

Install from source
-------------------

From a terminal, clone the repository and create an isolated environment:

.. code-block:: bash

   git clone https://github.com/ANTsX/ANTsNormalizingFlows.git
   cd ANTsNormalizingFlows
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

On Windows, replace the activation command with
``.venv\Scripts\activate`` in Command Prompt.
The editable installation uses the code in your checkout, so local source
changes are available without reinstalling the package.

For a particular accelerator configuration, install the appropriate PyTorch
build for your system before installing this package.

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import antsnormflows as nf; print(nf.__version__); print(nf.NormalizingFlow)"

The output should include a version and the ``NormalizingFlow`` class.
If the import fails, check that your terminal or notebook kernel uses the
same Python environment in which you installed the package.

Optional dependencies
---------------------

Run these commands from the repository root as needed:

.. code-block:: bash

   python -m pip install -e '.[examples]'
   python -m pip install -e '.[docs]'
   python -m pip install -e '.[test]'

The extras install notebook tools, Sphinx documentation tools, and testing
tools, respectively. Continue with :doc:`getting_started` for a complete
training example, or :doc:`examples` to explore the notebooks.
