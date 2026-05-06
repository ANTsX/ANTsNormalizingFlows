
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import flows


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _clamp_log_scale(log_scale, min_log=-5.0, max_log=5.0):
    # Hard clamp keeps sigma in [exp(min_log), exp(max_log)] ~ [6.7e-3, 148]
    if isinstance(log_scale, (float, int)):
        return torch.tensor(float(np.clip(log_scale, min_log, max_log)))
    return torch.clamp(log_scale, min_log, max_log)


def _apply_temperature(log_scale, temperature):
    if temperature is None:
        return log_scale
    # Multiplicative temperature on sigma => additive on log_sigma
    return log_scale + float(np.log(temperature))

# -----------------------------------------------------------------------------
# Base API
# -----------------------------------------------------------------------------

class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model.
    Parameters do not depend on the target variable (unlike VAE encoders).
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        raise NotImplementedError

    def log_prob(self, z):
        raise NotImplementedError

    def sample(self, num_samples=1, **kwargs):
        z, _ = self.forward(num_samples, **kwargs)
        return z


# -----------------------------------------------------------------------------
# Diagonal Gaussian (stable)
# -----------------------------------------------------------------------------

class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian with diagonal covariance.
    Adds clamped log-scales and safe numerics.
    """

    def __init__(self, shape, trainable=True, min_log=-5.0, max_log=5.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = int(np.prod(shape))
        self.min_log = float(min_log)
        self.max_log = float(max_log)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Optional multiplicative sigma factor

    def forward(self, num_samples=1, context=None, eps=None):
        if eps is None:
            eps = torch.randn(
                (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
            )
        else:
            if eps.shape != (num_samples,) + self.shape:
                raise ValueError(f"Expected eps shape {(num_samples,) + self.shape}, got {eps.shape}")

        log_scale = _apply_temperature(self.log_scale, self.temperature)
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)

        z = self.loc + torch.exp(log_scale) * eps
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1)))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        log_scale = _apply_temperature(self.log_scale, self.temperature)
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)
        inv_sigma = torch.exp(-log_scale)
        eps = (z - self.loc) * inv_sigma
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1)))
        )
        return log_p


# -----------------------------------------------------------------------------
# Conditional Diagonal Gaussian (stable)
# -----------------------------------------------------------------------------

class ConditionalDiagGaussian(BaseDistribution):
    """
    Conditional multivariate diagonal Gaussian;
    mean and log sigma are produced by a context encoder.
    """

    def __init__(self, shape, context_encoder, min_log=-5.0, max_log=5.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = int(np.prod(shape))
        self.context_encoder = context_encoder
        self.min_log = float(min_log)
        self.max_log = float(max_log)

    def forward(self, num_samples=1, context=None):
        enc = self.context_encoder(context)
        split_ind = enc.shape[-1] // 2
        mean = enc[..., :split_ind]
        log_scale = enc[..., split_ind:]
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)
        eps = torch.randn((num_samples,) + self.shape, dtype=mean.dtype, device=mean.device)
        z = mean + torch.exp(log_scale) * eps
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1)))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        enc = self.context_encoder(context)
        split_ind = enc.shape[-1] // 2
        mean = enc[..., :split_ind]
        log_scale = enc[..., split_ind:]
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)
        inv_sigma = torch.exp(-log_scale)
        eps = (z - mean) * inv_sigma
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1)))
        )
        return log_p


# -----------------------------------------------------------------------------
# Uniform (unchanged except for dtype/device safety)
# -----------------------------------------------------------------------------

class Uniform(BaseDistribution):
    """
    Multivariate uniform distribution over [low, high].
    """

    def __init__(self, shape, low=-1.0, high=1.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.d = int(np.prod(shape))
        self.register_buffer("low", torch.tensor(float(low)))
        self.register_buffer("high", torch.tensor(float(high)))
        # Cache scalar log_prob value
        self.log_prob_val = -self.d * np.log(float(high) - float(low))

    def forward(self, num_samples=1, context=None):
        eps = torch.rand((num_samples,) + self.shape, dtype=self.low.dtype, device=self.low.device)
        z = self.low + (self.high - self.low) * eps
        log_p = torch.full((num_samples,), self.log_prob_val, device=self.low.device)
        return z, log_p

    def log_prob(self, z, context=None):
        log_p = torch.full((z.shape[0],), self.log_prob_val, device=z.device, dtype=z.dtype)
        out_range = (z < self.low) | (z > self.high)
        ind_inf = torch.any(out_range.view(z.shape[0], -1), dim=-1)
        log_p[ind_inf] = -np.inf
        return log_p


# -----------------------------------------------------------------------------
# Uniform-Gaussian hybrid (minor broadcast fixes)
# -----------------------------------------------------------------------------

class UniformGaussian(BaseDistribution):
    """
    1D random variable with some entries uniform and others Gaussian.
    """

    def __init__(self, ndim, ind, scale=None):
        super().__init__()
        self.ndim = int(ndim)
        if isinstance(ind, int):
            ind = [ind]
        if torch.is_tensor(ind):
            self.register_buffer("ind", ind.to(dtype=torch.long))
        else:
            self.register_buffer("ind", torch.tensor(ind, dtype=torch.long))
        # Complement indices
        comp = [i for i in range(self.ndim) if i not in self.ind.tolist()]
        self.register_buffer("ind_", torch.tensor(comp, dtype=torch.long))
        # Permutations
        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.empty_like(perm_)
        inv_perm_[perm_] = torch.arange(self.ndim, device=perm_.device)
        self.register_buffer("inv_perm", inv_perm_)
        # Scales
        if scale is None:
            self.register_buffer("scale", torch.ones(self.ndim))
        else:
            self.register_buffer("scale", scale)

    def forward(self, num_samples=1, context=None):
        z = self.sample(num_samples)
        return z, self.log_prob(z)

    def sample(self, num_samples=1, context=None):
        device = self.scale.device
        dtype = self.scale.dtype
        eps_u = (torch.rand((num_samples, len(self.ind)), dtype=dtype, device=device) - 0.5)
        eps_g = torch.randn((num_samples, len(self.ind_)), dtype=dtype, device=device)
        z = torch.cat((eps_u, eps_g), dim=-1)
        z = z[..., self.inv_perm]
        return self.scale * z

    def log_prob(self, z, context=None):
        # Broadcast logs properly
        log_p_u = -torch.log(self.scale[self.ind]).sum().expand(z.shape[0])
        log_p_g = (
            -0.5 * len(self.ind_) * np.log(2 * np.pi)
            - torch.log(self.scale[self.ind_]).sum()
            - 0.5 * ((z[..., self.ind_] / self.scale[self.ind_]) ** 2).sum(dim=-1)
        )
        return log_p_u + log_p_g


# -----------------------------------------------------------------------------
# Class-conditional Diagonal Gaussian (stable)
# -----------------------------------------------------------------------------

class ClassCondDiagGaussian(BaseDistribution):
    """
    Class-conditional diagonal Gaussian with clamped log-scales.
    """

    def __init__(self, shape, num_classes, min_log=-5.0, max_log=5.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.perm = [self.n_dim] + list(range(self.n_dim))
        self.d = int(np.prod(shape))
        self.num_classes = int(num_classes)
        self.min_log = float(min_log)
        self.max_log = float(max_log)
        self.loc = nn.Parameter(torch.zeros(*self.shape, self.num_classes))
        self.log_scale = nn.Parameter(torch.zeros(*self.shape, self.num_classes))
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        if y is not None:
            num_samples = len(y)
        else:
            y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
        if y.dim() == 1:
            y_onehot = torch.zeros((num_samples, self.num_classes), dtype=self.loc.dtype, device=self.loc.device)
            y_onehot.scatter_(1, y[:, None], 1)
            y = y_onehot
        loc = (self.loc @ y.t()).permute(*self.perm)
        log_scale = (self.log_scale @ y.t()).permute(*self.perm)
        log_scale = _apply_temperature(log_scale, self.temperature)
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        z = loc + torch.exp(log_scale) * eps
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1)))
        )
        return z, log_p

    def log_prob(self, z, y):
        if y.dim() == 1:
            y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype, device=self.loc.device)
            y_onehot.scatter_(1, y[:, None], 1)
            y = y_onehot
        loc = (self.loc @ y.t()).permute(*self.perm)
        log_scale = (self.log_scale @ y.t()).permute(*self.perm)
        log_scale = _apply_temperature(log_scale, self.temperature)
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)
        inv_sigma = torch.exp(-log_scale)
        eps = (z - loc) * inv_sigma
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1)))
        )
        return log_p


# -----------------------------------------------------------------------------
# Glow-style channel-wise Gaussian (stable logs)
# -----------------------------------------------------------------------------

class GlowBase(BaseDistribution):
    """
    Diagonal Gaussian with one mean and log scale per channel (Glow).
    Uses clamped channel logs and safe numerics.
    """

    def __init__(self, shape, num_classes=None, logscale_factor=3.0, min_log=-5.0, max_log=5.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.num_pix = int(np.prod(shape[1:]))
        self.d = int(np.prod(shape))
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        self.logscale_factor = float(logscale_factor)
        self.min_log = float(min_log)
        self.max_log = float(max_log)

        self.loc = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.loc_logs = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.log_scale = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.log_scale_logs = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))

        if self.class_cond:
            self.loc_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
            self.log_scale_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))

        self.temperature = None

    def _prep_params(self, y=None):
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y is None:
                raise ValueError("y must be provided for class-conditional GlowBase.forward")
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype, device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0], *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0], *((self.n_dim - 1) * [1]))
        log_scale = _apply_temperature(log_scale, self.temperature)
        log_scale = _clamp_log_scale(log_scale, self.min_log, self.max_log)
        return loc, log_scale

    def forward(self, num_samples=1, y=None):
        if self.class_cond and (y is None):
            y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
        loc, log_scale = self._prep_params(y if self.class_cond else None)
        if not self.class_cond:
            num_samples = int(num_samples)
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        z = loc + torch.exp(log_scale) * eps
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - self.num_pix * torch.sum(log_scale, dim=self.sum_dim)
            - 0.5 * torch.sum(eps.pow(2), dim=self.sum_dim)
        )
        return z, log_p

    def log_prob(self, z, y=None):
        loc, log_scale = self._prep_params(y if self.class_cond else None)

        inv_sigma = torch.exp(-log_scale)
        eps = (z - loc) * inv_sigma
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - self.num_pix * torch.sum(log_scale, dim=self.sum_dim)
            - 0.5 * torch.sum(eps.pow(2), dim=self.sum_dim)
        )
        return log_p


# -----------------------------------------------------------------------------
# Affine Gaussian (delegates to flows; unchanged except dtype/device safety)
# -----------------------------------------------------------------------------

class AffineGaussian(BaseDistribution):
    """
    Diagonal Gaussian with an affine constant transform applied (optionally class conditional).
    """

    def __init__(self, shape, affine_shape, num_classes=None):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = int(np.prod(shape))
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.affine_shape = affine_shape
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        if self.class_cond:
            self.transform = flows.CCAffineConst(self.affine_shape, self.num_classes)
        else:
            self.transform = flows.AffineConstFlow(self.affine_shape)
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        dtype = self.transform.s.dtype
        device = self.transform.s.device
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=dtype, device=device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        log_scale = float(np.log(self.temperature)) if (self.temperature is not None) else 0.0
        eps = torch.randn((num_samples,) + self.shape, dtype=dtype, device=device)
        z = np.exp(log_scale) * eps
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - self.d * log_scale
            - 0.5 * torch.sum(eps.pow(2), dim=self.sum_dim)
        )
        if self.class_cond:
            z, log_det = self.transform(z, y)
        else:
            z, log_det = self.transform(z)
        log_p -= log_det
        return z, log_p

    def log_prob(self, z, y=None):
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.transform.s.dtype, device=self.transform.s.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        log_scale = float(np.log(self.temperature)) if (self.temperature is not None) else 0.0
        if self.class_cond:
            z, log_p = self.transform.inverse(z, y)
        else:
            z, log_p = self.transform.inverse(z)
        z = z / np.exp(log_scale)
        log_p = (
            log_p
            - self.d * log_scale
            - 0.5 * self.d * np.log(2 * np.pi)
            - 0.5 * torch.sum(z.pow(2), dim=self.sum_dim)
        )
        return log_p


# -----------------------------------------------------------------------------
# Gaussian Mixture (kept, with safe softmax)
# -----------------------------------------------------------------------------

class GaussianMixture(BaseDistribution):
    """
    Mixture of diagonal Gaussians.
    """

    def __init__(self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True, min_log=-5.0, max_log=5.0):
        super().__init__()
        self.n_modes = int(n_modes)
        self.dim = int(dim)
        self.min_log = float(min_log)
        self.max_log = float(max_log)

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1, keepdims=True)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(loc, dtype=torch.float32))
            self.log_scale = nn.Parameter(torch.tensor(np.log(scale), dtype=torch.float32))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(weights), dtype=torch.float32))
        else:
            self.register_buffer("loc", torch.tensor(loc, dtype=torch.float32))
            self.register_buffer("log_scale", torch.tensor(np.log(scale), dtype=torch.float32))
            self.register_buffer("weight_scores", torch.tensor(np.log(weights), dtype=torch.float32))

    def _clamped_logs(self):
        return _clamp_log_scale(self.log_scale, self.min_log, self.max_log)

    def forward(self, num_samples=1):
        weights = torch.softmax(self.weight_scores, dim=1)
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = F.one_hot(mode, self.n_modes)[..., None].to(dtype=self.loc.dtype, device=self.loc.device)

        eps = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        scale_sample = torch.sum(torch.exp(self._clamped_logs()) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps * scale_sample + loc_sample

        eps_all = (z[:, None, :] - self.loc) / torch.exp(self._clamped_logs())
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(eps_all.pow(2), 2)
            - torch.sum(self._clamped_logs(), 2)
        )
        log_p = torch.logsumexp(log_p, 1)
        return z, log_p

    def log_prob(self, z):
        weights = torch.softmax(self.weight_scores, dim=1)
        eps = (z[:, None, :] - self.loc) / torch.exp(self._clamped_logs())
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(eps.pow(2), 2)
            - torch.sum(self._clamped_logs(), 2)
        )
        log_p = torch.logsumexp(log_p, 1)
        return log_p


# -----------------------------------------------------------------------------
# Gaussian PCA (stabilized with Cholesky + slogdet)
# -----------------------------------------------------------------------------

class GaussianPCA(BaseDistribution):
    """
    Gaussian induced by a linear map of a latent (content) variable with Gaussian noise:
      z = W eps + loc,   eps ~ N(0, I_{latent_dim}),   noise ~ N(0, sigma^2 I)
    => z ~ N(loc,  W W^T + sigma^2 I).
    We implement stable sampling and log_prob using Cholesky and slogdet.
    """

    def __init__(self, dim, latent_dim=None, sigma=0.1, jitter=1e-6):
        super().__init__()
        self.dim = int(dim)
        self.latent_dim = int(latent_dim) if (latent_dim is not None) else int(dim)
        self.loc = nn.Parameter(torch.zeros(1, self.dim))
        # Initialize W with small scale to keep covariance well-conditioned at start
        self.W = nn.Parameter(0.05 * torch.randn(self.latent_dim, self.dim))
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma))))
        self.jitter = float(jitter)

    def _covariance(self):
        sigma2 = torch.exp(2.0 * self.log_sigma)
        I = torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        # Σ = Wᵀ W + σ² I  (shape: dim x dim)
        return self.W.t().mm(self.W) + sigma2 * I

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, self.latent_dim, dtype=self.loc.dtype, device=self.loc.device)
        z_ = eps.mm(self.W)                # (N x dim)
        z = z_ + self.loc                  # broadcast add
        # Log prob under N(loc, Σ)
        Sigma = self._covariance()
        # Stable Cholesky with jitter
        jitter = self.jitter
        for _ in range(5):
            try:
                L = torch.linalg.cholesky(Sigma + jitter * torch.eye(self.dim, dtype=Sigma.dtype, device=Sigma.device))
                break
            except RuntimeError:
                jitter *= 10.0
        # slogdet from Cholesky: logdet = 2 * sum(log(diag(L)))
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        # Quadratic form: solve L y = (z - loc)^T; then ||y||^2
        diff = z - self.loc
        # (N x dim) -> (dim x N) for triangular_solve expects (dim x N)
        y = torch.cholesky_solve(diff.t(), L)  # solves (L L^T) y = diff^T
        quad = 0.5 * torch.sum(diff.t() * y, dim=0)  # length-N
        log_p = -0.5 * self.dim * np.log(2 * np.pi) - 0.5 * logdet - quad
        return z, log_p

    def log_prob(self, z):
        diff = z - self.loc  # (N x dim)
        Sigma = self._covariance()
        jitter = self.jitter
        for _ in range(5):
            try:
                L = torch.linalg.cholesky(Sigma + jitter * torch.eye(self.dim, dtype=Sigma.dtype, device=Sigma.device))
                break
            except RuntimeError:
                jitter *= 10.0
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        y = torch.cholesky_solve(diff.t(), L)
        quad = 0.5 * torch.sum(diff.t() * y, dim=0)
        log_p = -0.5 * self.dim * np.log(2 * np.pi) - 0.5 * logdet - quad
        return log_p
