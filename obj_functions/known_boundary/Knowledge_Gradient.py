###########################################################################
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.utils.sampling import draw_sobol_normal_samples, manual_seed
from torch import Tensor
from torch.nn import Module
from torch.quasirandom import SobolEngine


class MCSampler(Module, ABC):
    r"""Abstract base class for Samplers.

    Subclasses must implement the `_construct_base_samples` method.

    Attributes:
        resample: If `True`, re-draw samples in each `forward` evaluation -
            this results in stochastic acquisition functions (and thus should
            not be used with deterministic optimization algorithms).
        collapse_batch_dims: If True, collapse the t-batch dimensions of the
            produced samples to size 1. This is useful for preventing sampling
            variance across t-batches.

    Example:
        This method is usually not called directly, but via the sampler's
        `__call__` method:
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(self, batch_range: Tuple[int, int] = (0, -2)) -> None:
        r"""Abstract base class for Samplers.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__()
        self.batch_range = batch_range
        self.register_buffer("base_samples", None)

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range."""
        return tuple(self._batch_range.tolist())

    @batch_range.setter
    def batch_range(self, batch_range: Tuple[int, int]):
        r"""Set the t-batch range and clear base samples.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        # set t-batch range if different; trigger resample & set base_samples to None
        if not hasattr(self, "_batch_range") or self.batch_range != batch_range:
            self.register_buffer(
                "_batch_range", torch.tensor(batch_range, dtype=torch.long)
            )
            self.register_buffer("base_samples", None)

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        base_sample_shape = self._get_base_sample_shape(posterior=posterior)
        self._construct_base_samples(posterior=posterior, shape=base_sample_shape)
        samples = posterior.rsample(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return samples

    def _get_base_sample_shape(self, posterior: Posterior) -> torch.Size:
        r"""Get the shape of the base samples.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The shape of the base samples expected by the posterior. If
            `collapse_batch_dims=True`, the t-batch dimensions of the base
            samples are collapsed to size 1. This is useful to prevent sampling
            variance across t-batches.
        """
        base_sample_shape = posterior.base_sample_shape
        if self.collapse_batch_dims:
            batch_start, batch_end = self.batch_range
            base_sample_shape = (
                base_sample_shape[:batch_start]
                + torch.Size([1 for _ in base_sample_shape[batch_start:batch_end]])
                + base_sample_shape[batch_end:]
            )
        return self.sample_shape + base_sample_shape

    @property
    def sample_shape(self) -> torch.Size:
        r"""The shape of a single sample."""
        return self._sample_shape

    @abstractmethod
    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate base samples (if necessary).

        This function will generate a new set of base samples and register the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has
            been adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        pass  # pragma: no cover
    



class SobolQMCNormalSampler(MCSampler):
    r"""Sampler for quasi-MC base samples using Sobol sequences.

    Example:
        >>> sampler = SobolQMCNormalSampler(1024, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(
        self,
        num_samples: int,
        resample: bool = False,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
        batch_range: Tuple[int, int] = (0, -2),
    ) -> None:
        r"""Sampler for quasi-MC base samples using Sobol sequences.

        Args:
            num_samples: The number of samples to use. As a best practice,
                use powers of 2.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__(batch_range=batch_range)
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate quasi-random Normal base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has been
            adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            self.resample
            or _check_shape_changed(self.base_samples, self.batch_range, shape)
            or (not self.collapse_batch_dims and shape != self.base_samples.shape)
        ):
            batch_start, batch_end = self.batch_range
            sample_shape, base_sample_shape = split_shapes(shape)
            output_dim = (
                base_sample_shape[:batch_start] + base_sample_shape[batch_end:]
            ).numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = draw_sobol_normal_samples(
                d=output_dim,
                n=(sample_shape + base_sample_shape[batch_start:batch_end]).numel(),
                device=posterior.device,
                dtype=posterior.dtype,
                seed=self.seed,
            )
            self.seed += 1
            base_samples = base_samples.view(shape)
            self.register_buffer("base_samples", base_samples)
        elif self.collapse_batch_dims and shape != posterior.base_sample_shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)
            
def _check_shape_changed(
    base_samples: Optional[Tensor], batch_range: Tuple[int, int], shape: torch.Size
) -> bool:
    r"""Check if the base samples shape matches a given shape in non batch dims.

    Args:
        base_samples: The Posterior for which to generate base samples.
        batch_range: The range t-batch dimensions to ignore for shape check.
        shape: The base sample shape to compare.

    Returns:
        A bool indicating whether the shape changed.
    """
    if base_samples is None:
        return True
    batch_start, batch_end = batch_range
    b_sample_shape, b_base_sample_shape = split_shapes(base_samples.shape)
    sample_shape, base_sample_shape = split_shapes(shape)
    return (
        b_sample_shape != sample_shape
        or b_base_sample_shape[batch_end:] != base_sample_shape[batch_end:]
        or b_base_sample_shape[:batch_start] != base_sample_shape[:batch_start]
    )



def split_shapes(
    base_sample_shape: torch.Size,
) -> Tuple[torch.Size, torch.Size]:
    r"""Split a base sample shape into sample and base sample shapes.

    Args:
        base_sample_shape: The base sample shape.

    Returns:
        A tuple containing the sample and base sample shape.
    """
    return base_sample_shape[:1], base_sample_shape[1:]



##################################################################################
import torch
import numpy as np
#from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP,FixedNoiseGP
from botorch.test_functions import Ackley,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,DixonPrice,Levy,StyblinskiTang,Griewank
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement,PosteriorMean,qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize,normalize
from torch.quasirandom import SobolEngine
from gpytorch.kernels import MaternKernel, RBFKernel, IndexKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means import ZeroMean
import matplotlib.pyplot as plt

#from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
#from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
#from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


#from __future__ import annotations

import math

from abc import ABC

from contextlib import nullcontext
from copy import deepcopy

from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.constants import get_constants_like
from botorch.utils.probability import MVNXPB
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi,
    log_prob_normal_in,
    ndtr as Phi,
    phi,
)
from botorch.utils.safe_math import log1mexp, logmeanexp
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor
from torch.nn.functional import pad

# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""
    Base class for analytic acquisition functions.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        if posterior_transform is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a "
                    "multi-output model."
                )
        else:
            if not isinstance(posterior_transform, PosteriorTransform):
                raise UnsupportedError(
                    "AnalyticAcquisitionFunctions only support PosteriorTransforms."
                )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )

    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the first and second moments of the model posterior.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            compute_sigma: Boolean indicating whether or not to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        self.to(device=X.device)  # ensures buffers / parameters are on the same device
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma

class DiscreteKnowledgeGradient(AnalyticAcquisitionFunction):
    r"""Knowledge Gradient using a fixed discretisation in the Design Space "X"."""

    def __init__(
        self,
        model: Model,
        bounds: Optional[Tensor] = None,
        num_discrete_points: Optional[int] = None,
        X_discretisation: Optional[Tensor] = None,
    ) -> None:
        r"""
        Discrete Knowledge Gradient
        Args:
            model: A fitted model.
            bounds: A `2 x d` tensor of lower and upper bounds for each column
            num_discrete_points: (int) The number of discrete points to use for input (X) space. More discrete
                points result in a better approximation, at the expense of
                memory and wall time.
            discretisation: A `k x d`-dim Tensor of `k` design points that will approximate the
                continuous space with a discretisation.
        """

        if X_discretisation is None:
            if num_discrete_points is None:
                raise ValueError(
                    "Must specify `num_discrete_points` for random discretisation if no `discretisation` is provided."
                )

            X_discretisation = draw_sobol_samples(
                bounds=bounds, n=num_discrete_points, q=1
            )

        super(AnalyticAcquisitionFunction, self).__init__(model=model)

        self.num_input_dimensions = bounds.shape[1]
        self.X_discretisation = X_discretisation

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xnew_idx, xnew in enumerate(X):
            xnew = xnew.unsqueeze(0)
            kgvals[xnew_idx] = self.compute_discrete_kg(
                xnew=xnew, optimal_discretisation=self.X_discretisation
            )
        return kgvals

    def compute_discrete_kg(
        self, xnew: Tensor, optimal_discretisation: Tensor
    ) -> Tensor:
        """

        Args:
        xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            optimal_discretisation: num_fantasies x d Tensor. Optimal X values for each z in zvalues.

        """
        # Augment the discretisation with the designs.
        concatenated_xnew_discretisation = torch.cat(
            [xnew, optimal_discretisation], dim=0
        ).squeeze()  # (m + num_X_disc, d)

        # Compute posterior mean, variance, and covariance.
        full_posterior = self.model.posterior(
            concatenated_xnew_discretisation, observation_noise=False
        )
        noise_variance = torch.unique(self.model.likelihood.noise_covar.noise)
        full_posterior_mean = full_posterior.mean  # (1 + num_X_disc , 1)

        # Compute full Covariante Cov(Xnew, X_discretised), select [Xnew X_discretised] submatrix, and subvectors.
        full_posterior_covariance = (
            full_posterior.mvn.covariance_matrix
        )  # (1 + num_X_disc , 1 + num_X_disc )
        posterior_cov_xnew_opt_disc = full_posterior_covariance[
            : len(xnew), :
        ].squeeze()  # ( 1 + num_X_disc,)
        full_posterior_variance = (
            full_posterior.variance.squeeze()
        )  # (1 + num_X_disc, )

        full_predictive_covariance = (
            posterior_cov_xnew_opt_disc
            / (full_posterior_variance + noise_variance).sqrt()
        )
        # initialise empty kgvals torch.tensor
        kgval = self.kgcb(a=full_posterior_mean, b=full_predictive_covariance)

        return kgval

    @staticmethod
    def kgcb(a: Tensor, b: Tensor) -> Tensor:
        r"""
        Calculates the linear epigraph, i.e. the boundary of the set of points
        in 2D lying above a collection of straight lines y=a+bx.
        Parameters
        ----------
        a
            Vector of intercepts describing a set of straight lines
        b
            Vector of slopes describing a set of straight lines
        Returns
        -------
        KGCB
            average height of the epigraph
        """

        a = a.squeeze()
        b = b.squeeze()
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"

        maxa = torch.max(a)

        if torch.all(torch.abs(b) < 0.000000001):
            return torch.Tensor([0])  # , np.zeros(a.shape), np.zeros(b.shape)

        # Order by ascending b and descending a. There should be an easier way to do this
        # but it seems that pytorch sorts everything as a 1D Tensor

        ab_tensor = torch.vstack([-a, b]).T
        ab_tensor_sort_a = ab_tensor[ab_tensor[:, 0].sort()[1]]
        ab_tensor_sort_b = ab_tensor_sort_a[ab_tensor_sort_a[:, 1].sort()[1]]
        a = -ab_tensor_sort_b[:, 0]
        b = ab_tensor_sort_b[:, 1]

        # exclude duplicated b (or super duper similar b)
        threshold = (b[-1] - b[0]) * 0.00001
        diff_b = b[1:] - b[:-1]
        keep = diff_b > threshold
        keep = torch.cat([torch.Tensor([True]), keep])
        keep[torch.argmax(a)] = True
        keep = keep.bool()  # making sure 0 1's are transformed to booleans

        a = a[keep]
        b = b[keep]

        # initialize
        idz = [0]
        i_last = 0
        x = [-torch.inf]

        n_lines = len(a)
        # main loop TODO describe logic
        # TODO not pruning properly, e.g. a=[0,1,2], b=[-1,0,1]
        # returns x=[-inf, -1, -1, inf], shouldn't affect kgcb
        while i_last < n_lines - 1:
            i_mask = torch.arange(i_last + 1, n_lines)
            x_mask = -(a[i_last] - a[i_mask]) / (b[i_last] - b[i_mask])

            best_pos = torch.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(torch.inf)

        x = torch.Tensor(x)
        idz = torch.LongTensor(idz)
        # found the epigraph, now compute the expectation
        a = a[idz]
        b = b[idz]

        normal = Normal(torch.zeros_like(x), torch.ones_like(x))

        pdf = torch.exp(normal.log_prob(x))
        cdf = normal.cdf(x)

        kg = torch.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        kg -= maxa
        return kg
   
   
    
#########################################################################################
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import Tensor

from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)

from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import (
    AcquisitionObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
# from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)


class MCKnowledgeGradient(DiscreteKnowledgeGradient):
    r"""Knowledge Gradient using Monte-Carlo integration.

    This computes the Knowledge Gradient using randomly generated Zj ~ N(0,1) to
    find \max_{x'}{ \mu^n(x') + \hat{\sigma^n{x', x}}Z_{j} }. Then, the outer
    expectation is solved by taking the Monte Carlo average.
    """

    def __init__(
            self,
            model: Model,
            num_fantasies: Optional[int] = 64,
            bounds: Tensor = None,
            inner_sampler: Optional[MCSampler] = None,
            objective: Optional[AcquisitionObjective] = None,
            seed: Optional[MCSampler] = 1,
            current_value: Optional[Tensor] = None,
            **kwargs: Any,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of
                the solutions to the inner problem.
            inner_sampler: The sampler used to sample fantasy observations.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
            kwargs: Additional keyword arguments. This includes the options for
                optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
                an `options` dictionary to be passed on to the optimization helpers, and
                a `scipy_options` dictionary to be passed to `scipy.minimize`.
        """

        if num_fantasies is None:
            raise ValueError("Must specify `num_fantasies`")

        super(MCKnowledgeGradient, self).__init__(
            model=model, bounds=bounds, num_discrete_points=num_fantasies
        )

        self.bounds = bounds
        self.dim = bounds.shape[1]
        self.num_fantasies = num_fantasies
        self.current_value = current_value
        self.inner_sampler = inner_sampler
        self.objective = objective
        self.inner_condition_sampler = kwargs.get("inner_condition_sampler", "random")
        self.num_restarts = kwargs.get("num_restarts", 20)
        self.raw_samples = kwargs.get("raw_samples", 1024)
        self.seed = seed
        self.num_X_observations = None
        self.kwargs = kwargs

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:

        zvalues = self.construct_z_vals(self.num_fantasies)
        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xnew_idx, xnew in enumerate(X):
            _, kgvals[xnew_idx] = self.compute_mc_kg(xnew=xnew, zvalues=zvalues)

        return kgvals

    def construct_z_vals(self, nz: int, device: Optional[torch.device] = None) -> Tensor:
        """make nz random z """

        current_number_of_observations = self.model.train_inputs[0].shape[0]

        # Z values are only updated if new data is included in the model.
        # This ensures that we can use a deterministic optimizer.

        if current_number_of_observations != self.num_X_observations:
            # This generates the fantasised samples according to a random seed.
            z_vals = draw_sobol_normal_samples(
                d=1,
                n=self.num_fantasies
            ).squeeze()  # 1 x num_fantasies
            self.z_vals = z_vals
            self.num_X_observations = current_number_of_observations

        else:
            z_vals = self.z_vals
        return z_vals.to(device=device)

    def compute_mc_kg(
            self, xnew: Tensor, zvalues: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            Zvals: 1 x num_fantasies Tensor with num_fantasies Normal quantiles.
        Returns:
            xstar_inner_optimisation: num_fantasies x d Tensor. Optimal X values for each z in Zvals.
            kg_estimated_value: 1 x 1 Tensor, Monte Carlo KG value of xnew
        """

        # There's a recurssion problem importing packages by one_shot kg. Hacky way of importing these packages.
        # TODO: find a better way to fix this
        if "gen_candidates_scipy" not in sys.modules:
            from botorch.generation.gen import gen_candidates_scipy
            from botorch.optim.initializers import gen_value_function_initial_conditions

        # Loop over xnew points
        fantasy_opt_val = torch.zeros((1, self.num_fantasies))  # 1 x num_fantasies
        xstar_inner_optimisation = torch.zeros((self.num_fantasies, xnew.shape[1]))

        # This setting makes sure that I can rewrite the base samples and use the quantiles.
        # Therefore, resample=False, collapse_batch_dims=True.
        sampler = SobolQMCNormalSampler(
            num_samples=1, resample=False, collapse_batch_dims=True
        )

        # loop over number of GP fantasised mean realisations
        for fantasy_idx in range(self.num_fantasies):

            # construct one realisation of the fantasy model by adding xnew. We rewrite the internal variable
            # base samples, such that the samples are taken from the quantile.
            zval = zvalues[fantasy_idx].view(1, 1, 1)
            sampler.base_samples = zval

            fantasy_model = self.model.fantasize(
                X=xnew, sampler=sampler, observation_noise=True
            )

            # get the value function and make sure gradients are enabled.
            with torch.enable_grad():
                value_function = _get_value_function(
                    model=fantasy_model,
                    objective=self.objective,
                    sampler=self.inner_sampler,
                    project=getattr(self, "project", None),
                )

                # optimize the inner problem
                if self.inner_condition_sampler == "random":
                    domain_offset = self.bounds[0]
                    domain_range = self.bounds[1] - self.bounds[0]
                    X_unit_cube_samples = torch.rand((self.raw_samples, 1, 1, self.dim))
                    X_initial_conditions_raw = X_unit_cube_samples * domain_range + domain_offset

                    mu_val_initial_conditions_raw = value_function.forward(X_initial_conditions_raw)
                    best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[:self.num_restarts]
                    X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]


                else:

                    X_initial_conditions = gen_value_function_initial_conditions(
                        acq_function=value_function,
                        bounds=self.bounds,
                        current_model=self.model,
                        num_restarts=self.num_restarts,
                        raw_samples=self.raw_samples,
                        options={
                            **self.kwargs.get("options", {}),
                            **self.kwargs.get("scipy_options", {}),
                        },
                    )

                x_value, value = gen_candidates_scipy(
                    initial_conditions=X_initial_conditions,
                    acquisition_function=value_function,
                    lower_bounds=self.bounds[0],
                    upper_bounds=self.bounds[1],
                    options=self.kwargs.get("scipy_options"),
                )
                x_value = x_value  # num initial conditions x 1 x d
                value = value.squeeze()  # num_initial conditions

                # find top x in case there are several initial conditions
                x_top = x_value[torch.argmax(value)]  # 1 x 1 x d

                # make sure to propagate kg gradients.
                with settings.propagate_grads(True):
                    x_top_val = value_function(X=x_top)

                fantasy_opt_val[:, fantasy_idx] = x_top_val
                xstar_inner_optimisation[fantasy_idx, :] = x_top.squeeze()

        kg_estimated_value = torch.mean(fantasy_opt_val, dim=-1)

        return xstar_inner_optimisation, kg_estimated_value


class HybridKnowledgeGradient(MCKnowledgeGradient):
    r"""Hybrid Knowledge Gradient using Monte-Carlo integration as described in
    Pearce M., Klaise J.,and Groves M. 2020. "Practical Bayesian Optimization
    of Objectives with Conditioning Variables". arXiv:2002.09996

    This acquisition function first computes high value design vectors using the
    predictive posterior GP mean for different Normal quantiles. Then discrete
    knowledge gradient is computed using the high value design vectors as a
    discretisation.
    """

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""

        Args:
            X: A `m x 1 x d` Tensor with `m` acquisition function evaluations of
            `d` dimensions. Currently DiscreteKnowledgeGradient does can't perform
            batched evaluations.

        Returns:
            kgvals: A 'm' Tensor with 'm' KG values.
        """

        """ compute hybrid KG """

        # generate equal quantile spaced z_vals
        zvalues = self.construct_z_vals(self.num_fantasies)

        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xnew_idx, xnew in enumerate(X):
            # Compute X discretisation using the different generated quantiles.
            x_star, _ = self.compute_mc_kg(xnew=xnew, zvalues=zvalues)

            # Compute value of discrete Knowledge Gradient using the generated discretisation
            kgvals[xnew_idx] = self.compute_discrete_kg(
                xnew=xnew, optimal_discretisation=x_star
            )

        return kgvals

    def construct_z_vals(self, nz: int, device: Optional[torch.device] = None) -> Tensor:
        """make nz equally quantile-spaced z values"""

        quantiles_z = (torch.arange(nz) + 0.5) * (1 / nz)
        normal = torch.distributions.Normal(0, 1)
        z_vals = normal.icdf(quantiles_z)
        return z_vals.to(device=device)
    



class ProjectedAcquisitionFunction(AcquisitionFunction):
    r"""
    Defines a wrapper around  an `AcquisitionFunction` that incorporates the project
    operator. Typically used to handle value functions in look-ahead methods.
    """

    def __init__(
            self,
            base_value_function: AcquisitionFunction,
            project: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(base_value_function.model)
        self.base_value_function = base_value_function
        self.project = project
        self.objective = base_value_function.objective
        self.sampler = getattr(base_value_function, "sampler", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.base_value_function(self.project(X))


def _get_value_function(
    model: Model,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    sampler: Optional[MCSampler] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    valfunc_cls: Optional[Type[AcquisitionFunction]] = None,
    valfunc_argfac: Optional[Callable[[Model], Dict[str, Any]]] = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if valfunc_cls is not None:
        common_kwargs: Dict[str, Any] = {
            "model": model,
            "posterior_transform": posterior_transform,
        }
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
            common_kwargs["objective"] = objective
        kwargs = valfunc_argfac(model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if objective is not None:
            base_value_function = qSimpleRegret(
                model=model,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
            )
        else:
            base_value_function = PosteriorMean(
                model=model, posterior_transform=posterior_transform
            )

    if project is None:
        return base_value_function
    else:
        return ProjectedAcquisitionFunction(
            base_value_function=base_value_function,
            project=project,
        )
###########################################################################
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.utils.sampling import draw_sobol_normal_samples, manual_seed
from torch import Tensor
from torch.nn import Module
from torch.quasirandom import SobolEngine


class MCSampler(Module, ABC):
    r"""Abstract base class for Samplers.

    Subclasses must implement the `_construct_base_samples` method.

    Attributes:
        resample: If `True`, re-draw samples in each `forward` evaluation -
            this results in stochastic acquisition functions (and thus should
            not be used with deterministic optimization algorithms).
        collapse_batch_dims: If True, collapse the t-batch dimensions of the
            produced samples to size 1. This is useful for preventing sampling
            variance across t-batches.

    Example:
        This method is usually not called directly, but via the sampler's
        `__call__` method:
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(self, batch_range: Tuple[int, int] = (0, -2)) -> None:
        r"""Abstract base class for Samplers.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__()
        self.batch_range = batch_range
        self.register_buffer("base_samples", None)

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range."""
        return tuple(self._batch_range.tolist())

    @batch_range.setter
    def batch_range(self, batch_range: Tuple[int, int]):
        r"""Set the t-batch range and clear base samples.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        # set t-batch range if different; trigger resample & set base_samples to None
        if not hasattr(self, "_batch_range") or self.batch_range != batch_range:
            self.register_buffer(
                "_batch_range", torch.tensor(batch_range, dtype=torch.long)
            )
            self.register_buffer("base_samples", None)

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        base_sample_shape = self._get_base_sample_shape(posterior=posterior)
        self._construct_base_samples(posterior=posterior, shape=base_sample_shape)
        samples = posterior.rsample(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return samples

    def _get_base_sample_shape(self, posterior: Posterior) -> torch.Size:
        r"""Get the shape of the base samples.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The shape of the base samples expected by the posterior. If
            `collapse_batch_dims=True`, the t-batch dimensions of the base
            samples are collapsed to size 1. This is useful to prevent sampling
            variance across t-batches.
        """
        base_sample_shape = posterior.base_sample_shape
        if self.collapse_batch_dims:
            batch_start, batch_end = self.batch_range
            base_sample_shape = (
                base_sample_shape[:batch_start]
                + torch.Size([1 for _ in base_sample_shape[batch_start:batch_end]])
                + base_sample_shape[batch_end:]
            )
        return self.sample_shape + base_sample_shape

    @property
    def sample_shape(self) -> torch.Size:
        r"""The shape of a single sample."""
        return self._sample_shape

    @abstractmethod
    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate base samples (if necessary).

        This function will generate a new set of base samples and register the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has
            been adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        pass  # pragma: no cover
    



class SobolQMCNormalSampler(MCSampler):
    r"""Sampler for quasi-MC base samples using Sobol sequences.

    Example:
        >>> sampler = SobolQMCNormalSampler(1024, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(
        self,
        num_samples: int,
        resample: bool = False,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
        batch_range: Tuple[int, int] = (0, -2),
    ) -> None:
        r"""Sampler for quasi-MC base samples using Sobol sequences.

        Args:
            num_samples: The number of samples to use. As a best practice,
                use powers of 2.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__(batch_range=batch_range)
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate quasi-random Normal base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has been
            adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            self.resample
            or _check_shape_changed(self.base_samples, self.batch_range, shape)
            or (not self.collapse_batch_dims and shape != self.base_samples.shape)
        ):
            batch_start, batch_end = self.batch_range
            sample_shape, base_sample_shape = split_shapes(shape)
            output_dim = (
                base_sample_shape[:batch_start] + base_sample_shape[batch_end:]
            ).numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = draw_sobol_normal_samples(
                d=output_dim,
                n=(sample_shape + base_sample_shape[batch_start:batch_end]).numel(),
                device=posterior.device,
                dtype=posterior.dtype,
                seed=self.seed,
            )
            self.seed += 1
            base_samples = base_samples.view(shape)
            self.register_buffer("base_samples", base_samples)
        elif self.collapse_batch_dims and shape != posterior.base_sample_shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)
            
def _check_shape_changed(
    base_samples: Optional[Tensor], batch_range: Tuple[int, int], shape: torch.Size
) -> bool:
    r"""Check if the base samples shape matches a given shape in non batch dims.

    Args:
        base_samples: The Posterior for which to generate base samples.
        batch_range: The range t-batch dimensions to ignore for shape check.
        shape: The base sample shape to compare.

    Returns:
        A bool indicating whether the shape changed.
    """
    if base_samples is None:
        return True
    batch_start, batch_end = batch_range
    b_sample_shape, b_base_sample_shape = split_shapes(base_samples.shape)
    sample_shape, base_sample_shape = split_shapes(shape)
    return (
        b_sample_shape != sample_shape
        or b_base_sample_shape[batch_end:] != base_sample_shape[batch_end:]
        or b_base_sample_shape[:batch_start] != base_sample_shape[:batch_start]
    )



def split_shapes(
    base_sample_shape: torch.Size,
) -> Tuple[torch.Size, torch.Size]:
    r"""Split a base sample shape into sample and base sample shapes.

    Args:
        base_sample_shape: The base sample shape.

    Returns:
        A tuple containing the sample and base sample shape.
    """
    return base_sample_shape[:1], base_sample_shape[1:]
