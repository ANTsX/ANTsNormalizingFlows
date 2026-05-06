# normflows

A PyTorch package (fork [source](https://github.com/VincentStimper/normalizing-flows)) 
for discrete normalizing flows (NICE, Real NVP, Glow, Neural Spline Flows, etc.).

## Installation

Requires Python ≥ 3.10 and a working PyTorch installation (GPU optional).

```bash
pip install -e .
```

To run example notebooks:

```bash
pip install -e .[examples]
```

## Quick start

```python
import antsnormflows as nf

# Base distribution (2D diagonal Gaussian)
base = nf.distributions.base.DiagGaussian(2)

# Real NVP with simple MLP conditioner
flows = []
num_layers = 8
for _ in range(num_layers):
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode="swap"))

model = nf.NormalizingFlow(base, flows)
loss = model.forward_kld(x)  # x: (batch, 2)
loss.backward()
```

## Documentation

* [Original documentation](https://vincentstimper.github.io/normalizing-flows/)

## Citation

If you use `antsnormflows`, please cite the corresponding paper:

> Stimper et al., (2023). normflows: A PyTorch Package for Normalizing Flows.
> Journal of Open Source Software, 8(86), 5361, https://doi.org/10.21105/joss.05361

**BibTeX**

```bibtex
@article{Stimper2023,
  author = {Vincent Stimper and David Liu and Andrew Campbell and Vincent Berenz and Lukas Ryll and Bernhard Schölkopf and José Miguel Hernández-Lobato},
  title = {normflows: A PyTorch Package for Normalizing Flows},
  journal = {Journal of Open Source Software},
  volume = {8},
  number = {86},
  pages = {5361},
  publisher = {The Open Journal},
  doi = {10.21105/joss.05361},
  url = {https://doi.org/10.21105/joss.05361},
  year = {2023}
}
```
