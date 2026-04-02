"""
GP model wrappers around GPy.

Three model families:
- IndependentGP  -- one GP per output, no information sharing
- ICM            -- Intrinsic Coregionalization Model (single shared kernel)
- LCM            -- Linear Coregionalization Model (Q independent latent GPs)

All built on GPy's GPCoregionalizedRegression with the util.multioutput helpers.
"""

import warnings
import numpy as np
import GPy

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Helper: augment test inputs with task index (required by GPy coregion models)
# ---------------------------------------------------------------------------

def _augment(X: np.ndarray, task_idx: int) -> np.ndarray:
    """Append a constant task-index column to X for GPy coregionalized prediction."""
    return np.hstack([X, np.full((len(X), 1), task_idx)])


def _noise_dict(task_idx: int, n: int) -> dict:
    """Return the Y_metadata dict that GPy coregion models expect at prediction."""
    return {"output_index": np.full((n, 1), task_idx, dtype=int)}


# ---------------------------------------------------------------------------
# 1.  Independent single-output GP
# ---------------------------------------------------------------------------

class IndependentGP:
    """
    Fit one GPRegression model per output using an ARD RBF kernel.

    Parameters
    ----------
    ARD : bool
        If True the RBF kernel learns a separate lengthscale per input dimension.
    n_restarts : int
        Number of random restarts for hyperparameter optimisation.
    """

    def __init__(self, ARD: bool = True, n_restarts: int = 5):
        self.ARD = ARD
        self.n_restarts = n_restarts
        self.models = []

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Fit one GP per column of Y_train.

        Parameters
        ----------
        X_train : (n, d)
        Y_train : (n, T)
        """
        self.models = []
        n_outputs = Y_train.shape[1]
        for t in range(n_outputs):
            y = Y_train[:, t : t + 1]
            k = GPy.kern.RBF(input_dim=X_train.shape[1], ARD=self.ARD)
            m = GPy.models.GPRegression(X_train, y, kernel=k)
            m.optimize_restarts(
                num_restarts=self.n_restarts, verbose=False, messages=False
            )
            self.models.append(m)
        return self

    def predict(self, X_new: np.ndarray):
        """
        Return (mean, var) each of shape (n, T).
        """
        means, vars_ = [], []
        for m in self.models:
            mu, v = m.predict(X_new)
            means.append(mu)
            vars_.append(v)
        return np.hstack(means), np.hstack(vars_)

    @property
    def nlml(self):
        """Sum of negative log marginal likelihoods across outputs."""
        return sum(-m.log_likelihood() for m in self.models)

    def lengthscales(self):
        """Return list of ARD lengthscale arrays, one per output."""
        return [m.kern.lengthscale.values.copy() for m in self.models]


# ---------------------------------------------------------------------------
# 2.  ICM  (Intrinsic Coregionalization Model)
# ---------------------------------------------------------------------------

class ICM:
    """
    Intrinsic Coregionalization Model — all outputs share a single RBF kernel,
    scaled by a learned T×T coregionalization matrix B = WW^T + diag(kappa).

    Parameters
    ----------
    W_rank : int
        Rank of W in B = WW^T + diag(kappa).
    ARD : bool
        Per-dimension lengthscales in the base RBF.
    n_restarts : int
        Random restarts for marginal-likelihood optimisation.
    """

    def __init__(self, W_rank: int = 1, ARD: bool = True, n_restarts: int = 5):
        self.W_rank = W_rank
        self.ARD = ARD
        self.n_restarts = n_restarts
        self.model = None
        self._num_outputs = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Parameters
        ----------
        X_train : (n, d)
        Y_train : (n, T)
        """
        self._num_outputs = Y_train.shape[1]
        d = X_train.shape[1]

        X_list = [X_train] * self._num_outputs
        Y_list = [Y_train[:, t : t + 1] for t in range(self._num_outputs)]

        base_k = GPy.kern.RBF(input_dim=d, ARD=self.ARD)
        icm_k = GPy.util.multioutput.ICM(
            input_dim=d,
            num_outputs=self._num_outputs,
            kernel=base_k,
            W_rank=self.W_rank,
        )
        self.model = GPy.models.GPCoregionalizedRegression(X_list, Y_list, kernel=icm_k)
        self.model.optimize_restarts(
            num_restarts=self.n_restarts, verbose=False, messages=False
        )
        return self

    def predict(self, X_new: np.ndarray):
        """Return (mean, var) each of shape (n, T)."""
        means, vars_ = [], []
        for t in range(self._num_outputs):
            X_aug = _augment(X_new, t)
            nd = _noise_dict(t, len(X_new))
            mu, v = self.model.predict(X_aug, Y_metadata=nd)
            means.append(mu)
            vars_.append(v)
        return np.hstack(means), np.hstack(vars_)

    @property
    def nlml(self):
        return -self.model.log_likelihood()

    def coregionalization_matrix(self) -> np.ndarray:
        """
        Return the T×T coregionalization matrix B = W W^T + diag(kappa).
        This encodes how strongly the outputs co-vary.
        """
        # The ICM kernel in GPy stores W and kappa inside the Coregion kern
        coregion_kern = self.model.kern.ICM0_B  # GPy internal naming
        return coregion_kern.B

    def _get_coregion(self):
        """Access the GPy Coregion kernel object (for inspecting W, kappa)."""
        # GPy names it <ICM_name>.coregion_kern or similar; we locate it robustly
        for part in self.model.kern.parts:
            for sub in getattr(part, "parts", [part]):
                if isinstance(sub, GPy.kern.src.coregionalize.Coregionalize):
                    return sub
        return None

    def mixing_weights(self) -> np.ndarray:
        """
        Return the W matrix (T × W_rank) of the coregionalization kernel.
        Columns correspond to latent processes; rows to outputs.
        """
        cg = self._get_coregion()
        if cg is not None:
            return cg.W.values.copy()
        return None


# ---------------------------------------------------------------------------
# 3.  LCM  (Linear Coregionalization Model)
# ---------------------------------------------------------------------------

class LCM:
    """
    Linear Coregionalization Model — Q latent GPs, each with its own RBF
    kernel, mixed into the outputs via a learned weight matrix W (T×Q).

    Parameters
    ----------
    num_latents : int
        Number of latent GPs (Q).
    W_rank : int
        Rank of each w_q w_q^T block (typically 1).
    ARD : bool
        Per-dimension lengthscales for each latent kernel.
    n_restarts : int
    """

    def __init__(
        self,
        num_latents: int = 2,
        W_rank: int = 1,
        ARD: bool = True,
        n_restarts: int = 5,
    ):
        self.num_latents = num_latents
        self.W_rank = W_rank
        self.ARD = ARD
        self.n_restarts = n_restarts
        self.model = None
        self._num_outputs = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Parameters
        ----------
        X_train : (n, d)
        Y_train : (n, T)
        """
        self._num_outputs = Y_train.shape[1]
        d = X_train.shape[1]

        X_list = [X_train] * self._num_outputs
        Y_list = [Y_train[:, t : t + 1] for t in range(self._num_outputs)]

        # One RBF kernel per latent — their hyperparameters are optimised independently,
        # so each latent can operate at a different length-scale.
        kernels = [
            GPy.kern.RBF(input_dim=d, ARD=self.ARD, name=f"rbf_q{q}")
            for q in range(self.num_latents)
        ]
        lcm_k = GPy.util.multioutput.LCM(
            input_dim=d,
            num_outputs=self._num_outputs,
            kernels_list=kernels,
            W_rank=self.W_rank,
            name="LCM",
        )
        self.model = GPy.models.GPCoregionalizedRegression(X_list, Y_list, kernel=lcm_k)
        self.model.optimize_restarts(
            num_restarts=self.n_restarts, verbose=False, messages=False
        )
        return self

    def predict(self, X_new: np.ndarray):
        """Return (mean, var) each of shape (n, T)."""
        means, vars_ = [], []
        for t in range(self._num_outputs):
            X_aug = _augment(X_new, t)
            nd = _noise_dict(t, len(X_new))
            mu, v = self.model.predict(X_aug, Y_metadata=nd)
            means.append(mu)
            vars_.append(v)
        return np.hstack(means), np.hstack(vars_)

    @property
    def nlml(self):
        return -self.model.log_likelihood()

    def _get_coregion_kernels(self):
        """Return list of Coregionalize kernel objects, one per latent."""
        cg_list = []
        for part in self.model.kern.parts:
            for sub in getattr(part, "parts", [part]):
                if isinstance(sub, GPy.kern.src.coregionalize.Coregionalize):
                    cg_list.append(sub)
        return cg_list

    def mixing_matrix(self) -> np.ndarray:
        """
        Return the T × Q mixing matrix W, where each column w_q is the mixing
        weight vector for the q-th latent GP.

        The (t, q) entry tells you how much output t loads onto latent q.
        """
        cg_list = self._get_coregion_kernels()
        if not cg_list:
            return None
        # Each coregion has a W of shape (T, W_rank); squeeze rank-1 to (T,)
        cols = []
        for cg in cg_list:
            w = cg.W.values.copy()
            cols.append(w[:, 0] if w.ndim > 1 and w.shape[1] == 1 else w)
        return np.column_stack(cols)  # (T, Q)

    def latent_lengthscales(self):
        """
        Return a (Q, d) array of learned ARD lengthscales, one row per latent.
        Useful for interpreting what each latent GP 'pays attention to'.
        """
        ls_list = []
        for part in self.model.kern.parts:
            rbf = None
            for sub in getattr(part, "parts", [part]):
                if isinstance(sub, GPy.kern.src.rbf.RBF):
                    rbf = sub
                    break
            if rbf is not None:
                ls_list.append(rbf.lengthscale.values.copy())
        return np.array(ls_list) if ls_list else None


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
        raise ValueError(f"Unknown model: {name}. Choose 'independent', 'icm', or 'lcm'.")
