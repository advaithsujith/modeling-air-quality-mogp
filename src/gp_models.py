"""
GP model wrappers around GPyTorch.

Three model families:
- IndependentGP  -- T independent exact GPs, batched over outputs (GPU-native)
- ICM            -- Intrinsic Coregionalization Model via MultitaskKernel
- LCM            -- Linear Coregionalization Model via LMCVariationalStrategy
                    (sparse/variational, scales to full dataset with inducing points)

All three classes expose the same public API as the original GPy wrappers:
    .fit(X_train, Y_train)
    .predict(X_new) -> (mean, var) both shape (n, T)
    .nlml           (scalar; ELBO-based for LCM)
    IndependentGP.lengthscales()        -> list of T arrays (d,)
    ICM.coregionalization_matrix()      -> (T, T)
    ICM.mixing_weights()                -> (T, W_rank)
    LCM.mixing_matrix()                 -> (T, Q)
    LCM.latent_lengthscales()           -> (Q, d)

ICM also exposes a .model compatibility shim so that script 03's
get_B_matrix() helper (which walks model.model.kern.parts looking for .B)
continues to work unchanged.
"""

import copy
import warnings
import numpy as np
import torch
import gpytorch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE  = torch.float32


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=_DTYPE, device=_DEVICE)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Random re-initialisation helper
# ---------------------------------------------------------------------------

def _random_reinit(model: torch.nn.Module) -> None:
    """Perturb all unconstrained parameters with N(0, 1) noise."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.5)


# ---------------------------------------------------------------------------
# ICM compatibility shim  (for script 03's get_B_matrix helper)
# ---------------------------------------------------------------------------

class _CoregionProxy:
    """Mimics a GPy Coregionalize kernel with a .B attribute."""
    def __init__(self, B: np.ndarray):
        self.B = B

class _KernProxy:
    def __init__(self, B: np.ndarray):
        self.parts = [_CoregionProxy(B)]

class _ModelProxy:
    """Exposed as icm.model so that model.model.kern.parts[0].B works."""
    def __init__(self, B: np.ndarray):
        self.kern = _KernProxy(B)


# ===========================================================================
# 1. IndependentGP
# ===========================================================================

class _BatchedExactGP(gpytorch.models.ExactGP):
    """T independent RBF GPs, batched along the first dimension."""
    def __init__(self, train_x, train_y, likelihood, d: int, T: int):
        super().__init__(train_x, train_y, likelihood)
        bs = torch.Size([T])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d, batch_shape=bs),
            batch_shape=bs,
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


class IndependentGP:
    """
    Fit one exact GP per output using an ARD RBF kernel.
    All T GPs are batched together for efficient GPU execution.

    Parameters
    ----------
    ARD : bool
        Per-dimension lengthscales (always True in this implementation).
    n_restarts : int
        Number of random restarts for hyperparameter optimisation.
    max_iter : int
        Adam iterations per restart.
    """

    def __init__(self, ARD: bool = True, n_restarts: int = 5, max_iter: int = 200):
        self.ARD = ARD
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self._gp = None
        self._likelihood = None
        self._T = None
        self._d = None
        self._nlml_val = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        n, d = X_train.shape
        T    = Y_train.shape[1]
        self._T = T
        self._d = d

        # GPyTorch batched GP expects Y of shape (T, n)
        X_t = _to_tensor(X_train)            # (n, d)
        Y_t = _to_tensor(Y_train.T)          # (T, n)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([T]),
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
        ).to(_DEVICE)
        model = _BatchedExactGP(X_t, Y_t, likelihood, d, T).to(_DEVICE)
        mll   = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        best_loss  = float("inf")
        best_model = None
        best_like  = None

        for restart in range(self.n_restarts):
            if restart > 0:
                _random_reinit(model)
                _random_reinit(likelihood)

            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()), lr=0.1
            )

            for _ in range(self.max_iter):
                optimizer.zero_grad()
                with gpytorch.settings.fast_computations():
                    out  = model(X_t)
                    loss = -mll(out, Y_t).sum()  # batched MLL gives (T,) -> scalar
                loss.backward()
                optimizer.step()

            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_model = copy.deepcopy(model.state_dict())
                best_like  = copy.deepcopy(likelihood.state_dict())

        model.load_state_dict(best_model)
        likelihood.load_state_dict(best_like)
        self._gp          = model
        self._likelihood  = likelihood
        self._nlml_val    = best_loss
        return self

    def predict(self, X_new: np.ndarray):
        self._gp.eval()
        self._likelihood.eval()
        X_t = _to_tensor(X_new)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._gp(X_t))
        mean = _to_numpy(pred.mean.T)      # (T, n) -> (n, T)
        var  = _to_numpy(pred.variance.T)
        return mean, var

    @property
    def nlml(self) -> float:
        return float(self._nlml_val)

    def lengthscales(self):
        """Return list of T ARD lengthscale arrays, each of shape (d,)."""
        ls = self._gp.covar_module.base_kernel.lengthscale  # (T, 1, d)
        return [_to_numpy(ls[t, 0]) for t in range(self._T)]


# ===========================================================================
# 2. ICM
# ===========================================================================

class _MultitaskExactGP(gpytorch.models.ExactGP):
    """Exact multi-task GP using MultitaskKernel (ICM structure)."""
    def __init__(self, train_x, train_y, likelihood, d: int, T: int, rank: int):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=T
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d),
            num_tasks=T,
            rank=rank,
        )

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


class ICM:
    """
    Intrinsic Coregionalization Model via GPyTorch's MultitaskKernel.
    Exploits Kronecker structure for O(n³ + T³) inference.

    Parameters
    ----------
    W_rank : int
        Rank of the task covariance factor W (B = WW^T + diag(kappa)).
    ARD : bool
        Per-dimension lengthscales in the base RBF.
    n_restarts : int
        Random restarts for marginal-likelihood optimisation.
    max_iter : int
        Adam iterations per restart.
    """

    def __init__(self, W_rank: int = 1, ARD: bool = True,
                 n_restarts: int = 5, max_iter: int = 300):
        self.W_rank    = W_rank
        self.ARD       = ARD
        self.n_restarts = n_restarts
        self.max_iter  = max_iter
        self._gp          = None
        self._likelihood  = None
        self._T           = None
        self._nlml_val    = None
        self.model        = None   # compatibility shim, set after fit()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        n, d = X_train.shape
        T    = Y_train.shape[1]
        self._T = T

        X_t = _to_tensor(X_train)   # (n, d)
        Y_t = _to_tensor(Y_train)   # (n, T)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=T,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
        ).to(_DEVICE)
        model = _MultitaskExactGP(X_t, Y_t, likelihood, d, T, self.W_rank).to(_DEVICE)
        mll   = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        best_loss  = float("inf")
        best_model = None
        best_like  = None

        for restart in range(self.n_restarts):
            if restart > 0:
                _random_reinit(model)
                _random_reinit(likelihood)

            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()), lr=0.1
            )

            for _ in range(self.max_iter):
                optimizer.zero_grad()
                with gpytorch.settings.fast_computations():
                    out  = model(X_t)
                    loss = -mll(out, Y_t)
                loss.backward()
                optimizer.step()

            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_model = copy.deepcopy(model.state_dict())
                best_like  = copy.deepcopy(likelihood.state_dict())

        model.load_state_dict(best_model)
        likelihood.load_state_dict(best_like)
        self._gp         = model
        self._likelihood = likelihood
        self._nlml_val   = best_loss
        # Refresh compatibility shim so script 03's get_B_matrix() works
        self.model = _ModelProxy(self.coregionalization_matrix())
        return self

    def predict(self, X_new: np.ndarray):
        self._gp.eval()
        self._likelihood.eval()
        X_t = _to_tensor(X_new)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._gp(X_t))
        mean = _to_numpy(pred.mean)      # (n, T)
        var  = _to_numpy(pred.variance)  # (n, T)
        return mean, var

    @property
    def nlml(self) -> float:
        return float(self._nlml_val)

    def coregionalization_matrix(self) -> np.ndarray:
        """Return the T×T coregionalization matrix B = W W^T + diag(kappa)."""
        task_k = self._gp.covar_module.task_covar_module
        W      = _to_numpy(task_k.covar_factor)   # (T, rank)
        kappa  = _to_numpy(task_k.var)             # (T,)
        return W @ W.T + np.diag(kappa)

    def mixing_weights(self) -> np.ndarray:
        """Return the W factor of shape (T, W_rank)."""
        return _to_numpy(self._gp.covar_module.task_covar_module.covar_factor)


# ===========================================================================
# 3. LCM
# ===========================================================================

class _LMCApproximateGP(gpytorch.models.ApproximateGP):
    """Sparse LCM via LMCVariationalStrategy with Q latent GPs."""
    def __init__(self, inducing_points, d: int, T: int, Q: int):
        # inducing_points: (Q, m, d)
        bs = torch.Size([Q])
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=bs
        )
        inner_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist,
            learn_inducing_locations=True,
        )
        var_strategy = gpytorch.variational.LMCVariationalStrategy(
            inner_strategy,
            num_tasks=T,
            num_latents=Q,
            latent_dim=-1,
        )
        super().__init__(var_strategy)
        self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d, batch_shape=bs),
            batch_shape=bs,
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


class LCM:
    """
    Linear Coregionalization Model — Q latent sparse GPs mixed into T outputs
    via a learned (T×Q) weight matrix. Uses variational inference with
    m inducing points so it scales to the full dataset.

    Parameters
    ----------
    num_latents : int
        Number of latent GPs (Q).
    W_rank : int
        Kept for API compatibility (not used; LMC handles mixing directly).
    ARD : bool
        Per-dimension lengthscales for each latent kernel.
    n_restarts : int
        Number of random restarts.
    num_inducing : int
        Number of inducing points per latent (m). Default 500.
    n_epochs : int
        Mini-batch training epochs per restart.
    batch_size : int
        Mini-batch size for ELBO computation.
    """

    def __init__(
        self,
        num_latents: int = 2,
        W_rank: int = 1,
        ARD: bool = True,
        n_restarts: int = 2,
        num_inducing: int = 500,
        n_epochs: int = 300,
        batch_size: int = 256,
    ):
        self.num_latents  = num_latents
        self.W_rank       = W_rank
        self.ARD          = ARD
        self.n_restarts   = n_restarts
        self.num_inducing = num_inducing
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self._gp          = None
        self._likelihood  = None
        self._T           = None
        self._d           = None
        self._X_t         = None
        self._Y_t         = None
        self._nlml_val    = None

    def _init_inducing(self, X_train: np.ndarray) -> torch.Tensor:
        """K-Means inducing points, replicated for each latent: (Q, m, d)."""
        m = min(self.num_inducing, len(X_train))
        km = MiniBatchKMeans(n_clusters=m, n_init=3, random_state=0)
        km.fit(X_train)
        centers = km.cluster_centers_.astype(np.float32)   # (m, d)
        z = torch.tensor(centers, dtype=_DTYPE, device=_DEVICE)
        return z.unsqueeze(0).expand(self.num_latents, -1, -1).clone()

    def _compute_elbo(self, model, likelihood, mll, X_t, Y_t) -> float:
        model.eval(); likelihood.eval()
        with torch.no_grad():
            out  = model(X_t)
            loss = -mll(out, Y_t)
        return loss.item()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        n, d = X_train.shape
        T    = Y_train.shape[1]
        self._T = T
        self._d = d

        X_t = _to_tensor(X_train)   # (n, d)
        Y_t = _to_tensor(Y_train)   # (n, T)
        self._X_t = X_t
        self._Y_t = Y_t

        dataset    = TensorDataset(X_t, Y_t)
        loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        inducing_pts = self._init_inducing(X_train)   # (Q, m, d)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=T,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
        ).to(_DEVICE)
        model = _LMCApproximateGP(inducing_pts, d, T, self.num_latents).to(_DEVICE)
        mll   = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

        best_loss  = float("inf")
        best_model = None
        best_like  = None

        for restart in range(self.n_restarts):
            if restart > 0:
                # Re-initialise inducing locations and parameters
                new_ip = self._init_inducing(X_train)
                model.variational_strategy.base_variational_strategy.inducing_points.data.copy_(new_ip)
                _random_reinit(model)
                _random_reinit(likelihood)

            model.train(); likelihood.train()
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()), lr=0.01
            )

            for epoch in range(self.n_epochs):
                for X_batch, Y_batch in loader:
                    optimizer.zero_grad()
                    out  = model(X_batch)
                    loss = -mll(out, Y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluate ELBO on full training set for restart comparison
            final_loss = self._compute_elbo(model, likelihood, mll, X_t, Y_t)
            if final_loss < best_loss:
                best_loss  = final_loss
                best_model = copy.deepcopy(model.state_dict())
                best_like  = copy.deepcopy(likelihood.state_dict())

        model.load_state_dict(best_model)
        likelihood.load_state_dict(best_like)
        self._gp         = model
        self._likelihood = likelihood
        self._nlml_val   = best_loss
        return self

    def predict(self, X_new: np.ndarray):
        self._gp.eval()
        self._likelihood.eval()
        X_t = _to_tensor(X_new)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._gp(X_t))
        mean = _to_numpy(pred.mean)      # (n, T)
        var  = _to_numpy(pred.variance)  # (n, T)
        return mean, var

    @property
    def nlml(self) -> float:
        """Negative ELBO (lower = better, comparable within experiment)."""
        return float(self._nlml_val)

    def mixing_matrix(self) -> np.ndarray:
        """
        Return the T × Q mixing matrix W.
        lmc_coefficients is stored as (Q, T) in GPyTorch; transposed to (T, Q).
        """
        lmc = self._gp.variational_strategy.lmc_coefficients   # (Q, T)
        W   = _to_numpy(lmc)
        if W.shape[0] == self.num_latents:
            W = W.T   # -> (T, Q)
        return W

    def latent_lengthscales(self) -> np.ndarray:
        """Return (Q, d) ARD lengthscales, one row per latent GP."""
        ls = self._gp.covar_module.base_kernel.lengthscale   # (Q, 1, d)
        return _to_numpy(ls[:, 0, :])


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_model(name: str, **kwargs):
    """
    Factory for building models by name string.

    Parameters
    ----------
    name : 'independent', 'icm', 'lcm'
    kwargs : passed to the respective class constructor
    """
    name = name.lower()
    if name == "independent":
        return IndependentGP(**kwargs)
    elif name == "icm":
        return ICM(**kwargs)
    elif name == "lcm":
        return LCM(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose 'independent', 'icm', or 'lcm'.")
