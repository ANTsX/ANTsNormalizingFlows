Getting started
===============

This example trains a two-dimensional Real NVP model on a synthetic mixture
of Gaussians. It includes data generation, optimization, density evaluation,
and sampling. No downloaded dataset is needed.

Build and train a model
-----------------------

Run the following block as a Python script or in a notebook after completing
:doc:`installation`:

.. code-block:: python

   import torch
   import antsnormflows as nf

   torch.manual_seed(42)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   def make_model():
       base = nf.distributions.base.DiagGaussian(2)
       flows = []
       for _ in range(4):
           conditioner = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
           flows.append(nf.flows.AffineCouplingBlock(conditioner))
           flows.append(nf.flows.Permute(2, mode="swap"))
       return nf.NormalizingFlow(base, flows)

   def draw_data(n):
       centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], device=device)
       labels = torch.randint(2, (n,), device=device)
       return centers[labels] + 0.5 * torch.randn(n, 2, device=device)

   model = make_model().to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   model.train()
   for step in range(200):
       x = draw_data(256)
       optimizer.zero_grad(set_to_none=True)
       loss = model.forward_kld(x)
       if not torch.isfinite(loss):
           raise RuntimeError("Non-finite training loss")
       loss.backward()
       optimizer.step()
       if (step + 1) % 50 == 0:
           print(f"Step {step + 1}: NLL = {loss.item():.3f}")

Each coupling layer leaves one coordinate unchanged and uses it to predict
the scale and shift of the other coordinate. The conditioner therefore has
one input and two outputs. Permutations exchange the coordinates so that
both can be transformed across successive layers.

``forward_kld(x)`` returns the mean negative log-likelihood of the batch.
Minimizing it fits the model to the observations; its value can be negative
because a continuous probability density can exceed one. This short run is
a demonstration, not a convergence guarantee.

Evaluate and sample
-------------------

Continue in the same Python session:

.. code-block:: python

   model.eval()
   with torch.no_grad():
       validation = draw_data(1024)
       log_prob = model.log_prob(validation)
       samples, sample_log_prob = model.sample(512)
       z = model.inverse(validation)
       reconstructed = model(z)

   print("Validation NLL:", -log_prob.mean().item())
   print("Sample shape:", samples.shape)  # (512, 2)
   print("Log-density shape:", sample_log_prob.shape)  # (512,)
   print("Round-trip error:", (validation - reconstructed).abs().max().item())

``log_prob`` gives one log density per observation, summed over its features.
``sample`` returns both the generated observations and their model log
densities. ``model(z)`` maps latent coordinates to data, whereas
``model.inverse(x)`` maps data to latent coordinates. Numerical round-trip
errors depend on the learned transformation and floating-point precision.

Save and reload
---------------

Continue with the same ``make_model`` function:

.. code-block:: python

   model.save("real_nvp.pt")
   restored = make_model()
   restored.load("real_nvp.pt", map_location="cpu")
   restored.eval()
   with torch.no_grad():
       restored_samples, _ = restored.sample(16)

The checkpoint contains the model's state dictionary, including learned
parameters and registered buffers. Recreate the same architecture before
loading it. This convenience method does not save the optimizer or training
step; see :doc:`training` for a resumable checkpoint.
