Examples
========

The example notebooks can be viewed on GitHub or downloaded and run locally
after installing the example dependencies with ``pip install -e '.[examples]'``.

Run locally
-----------

From the repository root, launch JupyterLab using the environment where you
installed the package:

.. code-block:: bash

   python -m pip install -e '.[examples]'
   python -m jupyterlab examples

Select that environment's kernel and run the notebook cells in order.
Some examples download datasets or use optional accelerators; inspect their
setup and data-loading cells before running a long experiment.

Suggested learning path
-----------------------

#. Start with :doc:`getting_started` for a self-contained, synthetic-data
   example with training, evaluation, and checkpoints.
#. Explore Real NVP to study affine coupling and density fitting.
#. Compare planar, radial, and affine flows to explore different transforms.
#. Try neural spline flows for more flexible transformations, then the
   conditional example for context-dependent distributions.
#. Explore Glow and image flows for multiscale image architectures, or the
   variational autoencoder example for latent-variable models.

Notebook compatibility
----------------------

These notebooks include examples inherited from the upstream project and
may require adaptation to the current API. In particular, the current
package exports ``GlowBlock2d`` / ``GlowBlock3d`` and
``Squeeze2d`` / ``Squeeze3d``. Older cells using ``GlowBlock`` or ``Squeeze``
need the dimension-specific name (for example, ``Squeeze2d`` in a 2D image
example). Use ``import antsnormflows as nf`` for this package and consult
:doc:`api` for current signatures. The self-contained tutorial above is the
recommended first check of your installation.

Normalizing flows
-----------------

* `Augmented flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/augmented_flow.ipynb>`_
* `Change the base distribution <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/change_base_distribution.ipynb>`_
* `Circular neural spline flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/circular_nsf.ipynb>`_
* `Compare planar, radial, and affine flows <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/comparison_plan_rad_aff.ipynb>`_
* `Conditional flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/conditional_flow.ipynb>`_
* `Glow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/glow.ipynb>`_
* `Image flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/image.ipynb>`_
* `Neural spline flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/neural_spline_flow.ipynb>`_
* `Paper example: neural spline flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/paper_example_nsf.ipynb>`_
* `Planar flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/planar.ipynb>`_
* `Real NVP <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/real_nvp.ipynb>`_
* `Residual flow <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/residual.ipynb>`_

Variational autoencoders
------------------------

* `Variational autoencoder <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/vae.ipynb>`_

Colab variants
--------------

* `Glow (Colab) <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/glow_colab.ipynb>`_
* `Paper example (Colab) <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/paper_example_nsf_colab.ipynb>`_
* `Real NVP (Colab) <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/real_nvp_colab.ipynb>`_
