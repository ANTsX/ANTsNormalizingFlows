Building the documentation
==========================

From the repository root, install the documentation dependencies and build
the HTML site:

.. code-block:: bash

   python -m pip install -e '.[docs]'
   python -m sphinx -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. For a clean build that
treats warnings as errors, use:

.. code-block:: bash

   python -m sphinx -E -a -W --keep-going -b html docs docs/_build/html

The documentation is written in reStructuredText. Add new pages to the
``toctree`` in ``docs/index.rst`` so they appear in the navigation.
The API reference is generated from Python docstrings using autodoc and
autosummary; it imports the package, so its runtime dependencies must be
installed as well.

Read the Docs uses ``.readthedocs.yaml`` to select Python, install the package
with its ``docs`` extra, and build ``docs/conf.py``. Local edits become part
of the hosted documentation only after they are committed, pushed to the
configured repository, and a Read the Docs build succeeds.
