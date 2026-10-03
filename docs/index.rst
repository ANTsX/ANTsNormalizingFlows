ANTsNormalizingFlows documentation
==================================

ANTsNormalizingFlows is an updated PyTorch package for discrete normalizing
flows, based on the `normflows project
<https://github.com/VincentStimper/normalizing-flows>`_.

.. image:: https://raw.githubusercontent.com/ANTsX/ANTsNormalizingFlows/coverage-badge/coverage.svg
   :alt: Test coverage
   :target: https://github.com/ANTsX/ANTsNormalizingFlows/actions/workflows/pytest.yaml

Start here
----------

Follow :doc:`installation`, then run :doc:`getting_started` to train a Real
NVP model on synthetic data. The :doc:`training` guide explains objectives,
tensor shapes, model conventions, and checkpointing. Browse :doc:`examples`
for larger experiments and :doc:`api` for class and method details.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   getting_started
   training
   examples
   api
   development


Citations
---------

If you use ANTsNormalizingFlows in your research, please cite the following papers:

Tustison et al. (2026). *Deep Computational Anatomy via Latent-Aligned Normalizing Flows*. bioRxiv.

.. dropdown:: BibTeX

   .. code-block:: bibtex

      @article {Tustison2026.05.05.723039,
        author = {Tustison, Nicholas J. and Avants, Brian B. and Cook, Philip A. and Gee, James C. and Stone, James R.},
        title = {Deep Computational Anatomy via Latent-Aligned Multiview Normalizing Flows},
        elocation-id = {2026.05.05.723039},
        year = {2026},
        doi = {10.64898/2026.05.05.723039},
        URL = {https://www.biorxiv.org/content/early/2026/05/11/2026.05.05.723039},
        eprint = {https://www.biorxiv.org/content/early/2026/05/11/2026.05.05.723039.full.pdf},
        journal = {bioRxiv}
      }

Stimper et al. (2023). *normflows: A PyTorch Package for Normalizing Flows*. Journal of Open Source Software, 8(86), 5361.

.. dropdown:: BibTeX

   .. code-block:: bibtex

      @article{Stimper2023,
        doi = {10.21105/joss.05361},
        url = {https://doi.org/10.21105/joss.05361},
        year = {2023},
        publisher = {The Open Journal},
        volume = {8}, number = {86},
        pages = {5361},
        author = {Stimper, Vincent and Liu, David and Campbell, Andrew and Berenz, Vincent and Ryll, Lukas and Schölkopf, Bernhard and Hernández-Lobato, José Miguel},
        title = {normflows: A PyTorch Package for Normalizing Flows},
        journal = {Journal of Open Source Software},
      }


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
