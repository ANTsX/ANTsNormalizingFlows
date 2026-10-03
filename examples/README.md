# Examples

Install from the repository root and launch JupyterLab:

```bash
python -m pip install -e '.[examples]'
python -m jupyterlab examples
```

Use the same Python environment as the notebook kernel. Run cells in order.
The new convolutional examples require the current source checkout.

## New convolutional examples

| Notebook | Main features | Data |
| --- | --- | --- |
| [conv_flow_1d.ipynb](conv_flow_1d.ipynb) | Coupling/Glow 1D, circular padding, temperature, checkpoint reload | Synthetic periodic signals |
| [glow_3d.ipynb](glow_3d.ipynb) | Two-level Glow 3D, checkpointed training, latent interpolation, shape cache | Synthetic noisy volumes |
| [factorized_glow_3d.ipynb](factorized_glow_3d.ipynb) | Per-axis squeezing, spatial/temporal kernels, identity initialization | Synthetic moving blobs |

These notebooks have short CPU-compatible training loops, finite-loss checks,
round-trip checks, and density consistency checks. They illustrate APIs, not
converged scientific models. CUDA is selected if available. Checkpoints use
temporary directories; replace them with a persistent path to retain results.

For a headless run, install the examples extra above, then run from the
repository root (write executed notebooks outside the source directory):

```bash
python -m jupyter nbconvert --to notebook --execute examples/conv_flow_1d.ipynb --output-dir /tmp/antsnf-executed --ExecutePreprocessor.timeout=600
```

Replace the filename to execute either 3D notebook.

## Existing experiments

- `real_nvp`, `planar`, `residual`, and `neural_spline_flow`: vector density models.
- `conditional_flow`: context-conditioned affine and spline models.
- `change_base_distribution`, `augmented_flow`, and `circular_nsf`: alternative latent spaces.
- `paper_example_nsf`: spline-flow figures; `comparison_plan_rad_aff` compares flow families and writes `.npz` results in the working directory.
- `image`: fits a synthetic image density; replace the image array with your own nonnegative grayscale image.
- `glow`: class-conditional 2D Glow; downloads CIFAR-10.
- `vae`: flow VAE; downloads MNIST.
- `*_colab`: source-installing variants for Colab.

The original research notebooks retain their longer training defaults. Reduce
iteration counts, model depth, and plotting sample counts for smoke checks.
Outputs are cleared so checked-in plots cannot be mistaken for current results.

## Command-line VAE scripts

```bash
python examples/vae.py --dataset mnist --flow Planar --epochs 1 --no-cuda
python examples/plain_vae.py --dataset mnist --epochs 1 --no-cuda
```

Both scripts download only the requested dataset. CIFAR channels are modeled
as separate observations in these legacy architectures. Repeated experiments
use `--experiment_mode --runs 3` and write CSV summaries in `experiments/`.

## Current API details

- Use `GlowBlock2d` and `Squeeze2d` for 2D images.
- `Squeeze3d.inverse` packs spatial dimensions into channels; `forward` unpacks them.
- Before `MultiscaleFlow.sample`, call `inverse_and_log_det` on a representative batch, including after checkpoint reload. `log_prob` alone does not populate the cache.
- Class-conditional sampling requires `num_samples=len(y)` when supplying a label batch.
- Multiscale gradient checkpointing is used by the transform-and-log-determinant methods in training mode. The 3D notebook builds its loss from that path.
- Glow activation clipping is an emergency safeguard; if it activates, exact inversion and likelihood interpretation are no longer guaranteed.
