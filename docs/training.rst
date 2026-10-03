Training and model conventions
==============================

Choosing an objective
---------------------

For observed data, use ``model.forward_kld(x)`` to minimize mean negative
log-likelihood. It corresponds to forward KL minimization up to a constant
that does not depend on the model parameters. All layers along this path
must implement an inverse and its log-determinant.

If you have a target density instead of a dataset, construct
``NormalizingFlow(base, flows, p=target)`` and use
``model.reverse_kld(num_samples=256)``. The target must supply
``log_prob(x)``. With the default ``beta=1``, this estimates the mean of
``log q(x) - log p(x)`` for model samples. An unnormalized target adds an
unknown constant to the reported objective. Gradients through the target
must be available for the usual differentiable training path.

Not every layer supports both directions. For example, ``Radial`` does not
implement an inverse, and ``Planar`` supports an inverse only for its
leaky-ReLU variant. Choose layers compatible with the objective and density
evaluation you need.

Shapes and devices
------------------

* Vector observations use ``(batch, features)``.
* Two-dimensional convolutional flows use ``(batch, channels, height, width)``.
* Three-dimensional convolutional flows use
  ``(batch, channels, depth, height, width)``.
* Put the model, data, and any conditioning tensors on the same device,
  with compatible floating-point dtypes.

Use finite floating-point inputs. Preprocessing is part of the density
model: a change of units or a nonlinear transform changes the density and
may require a Jacobian correction when reporting likelihoods in original
units. Discrete image intensities also require an explicit modeling choice,
such as dequantization, before being treated as continuous observations.

Model interfaces
----------------

.. list-table:: Common entry points
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Calling convention
   * - ``NormalizingFlow``
     - ``model(z)`` transforms latent coordinates to observations.
   * - ``ConditionalNormalizingFlow``
     - ``model(z, context=context)`` transforms latent coordinates with context.
   * - ``MultiscaleFlow``
     - ``model(x)`` returns a vector of negative log-likelihoods, one per observation.

For ``MultiscaleFlow``, use ``model.forward_kld(x)`` or ``model(x).mean()``
for a scalar training objective. Its base distributions and flow lists must
have the same number of levels, with one fewer merge operation than levels.
Set ``class_cond=False`` when constructing an unconditional multiscale model;
the default is ``True``.

Conditional models require a compatible base distribution and flow layers
that accept the context argument. Use matching batches of observations and
context, and supply context consistently for training, evaluation, and
sampling. See the conditional notebook in :doc:`examples`.

Validation and numerical issues
-------------------------------

Train with ``model.train()`` and evaluate with ``model.eval()`` and
``torch.no_grad()``. Monitor mean negative log-likelihood on held-out data,
inspect generated samples, and check inverse/forward reconstruction where
both directions are supported.

If training produces non-finite values, check the input data and loss before
updating parameters. Try a lower learning rate, inspect predicted scales,
and establish a full-precision baseline before using mixed precision.
Data-dependent normalization layers such as ActNorm need a representative
initial batch; retain their initialized buffers when saving a model.

Resuming training
-----------------

The following continues the example in :doc:`getting_started` and also saves
the optimizer state:

.. code-block:: python

   torch.save({
       "model": model.state_dict(),
       "optimizer": optimizer.state_dict(),
       "step": step + 1,
   }, "training.pt")

   checkpoint = torch.load("training.pt", map_location=device, weights_only=True)
   model = make_model().to(device)
   model.load_state_dict(checkpoint["model"])
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   optimizer.load_state_dict(checkpoint["optimizer"])
   start_step = checkpoint["step"]
   model.train()

Resume your optimization loop from ``start_step``. Record the architecture,
preprocessing, and package versions alongside the checkpoint. Exact
reproduction of a stochastic run additionally requires random-generator
states and data-loader state; the minimal checkpoint above does not include
those.
