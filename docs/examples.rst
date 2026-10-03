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

Recent convolutional features
-----------------------------

These short notebooks use synthetic data and need no dataset downloads.
Install the current source checkout to use the latest APIs.

* `Convolutional 1D flows <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/conv_flow_1d.ipynb>`_:
  periodic signals, coupling versus Glow blocks, circular padding,
  temperature sampling, and checkpoints.
* `Multiscale Glow 3D <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/glow_3d.ipynb>`_:
  synthetic volumes, latent shapes, checkpointed training, reconstruction,
  interpolation, and reloading.
* `Factorized Glow 3D <https://github.com/ANTsX/ANTsNormalizingFlows/blob/main/examples/factorized_glow_3d.ipynb>`_:
  per-axis squeezing, spatial/temporal kernels, temporal identity
  initialization, and comparison with isotropic kernels.

The multiscale example explicitly uses ``inverse_and_log_det`` to populate
latent shapes before sampling and to exercise gradient checkpointing.
In the current implementation, ``log_prob`` and ``forward_kld`` do not use
that checkpointing path or populate the sampling shape cache.

The existing notebooks use the dimension-specific APIs, including
``GlowBlock2d`` and ``Squeeze2d``. Colab installation cells install the GitHub
source. Glow and VAE notebooks download CIFAR-10 and MNIST, respectively;
the image-density notebook uses a synthetic image by default. Long-running
original experiments retain their training budgets, so reduce them for a
quick check. Notebook outputs are cleared to avoid displaying stale results.

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

Command-line VAE experiments
----------------------------

The ``examples/vae.py`` and ``examples/plain_vae.py`` scripts provide flow
and baseline VAE experiments. For example, from the repository root:

.. code-block:: bash

   python examples/vae.py --dataset mnist --flow Planar --epochs 1 --no-cuda
   python examples/plain_vae.py --dataset mnist --epochs 1 --no-cuda

Each script downloads only the selected dataset. CIFAR images are treated
as separate channel observations by these legacy VAE architectures, not as
a joint RGB density. ``--experiment_mode --runs 3`` repeats an experiment
and writes a CSV summary under ``experiments/``.
