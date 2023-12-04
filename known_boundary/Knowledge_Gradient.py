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
    
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
    
##################### GP acquisition function ########################################################
def PosteriorMean(X,dim,f_best,model,f_star='no'): # X is a 2-dimensional array because we will use it in scipy.minimize

  X = X.reshape(-1,dim)
  
  mean,var = model.predict_noiseless(X)
  
  
  return mean.ravel()  #make the shape to be 1 dimensional



def PosteriorMean_acquisition_opt(model,bounds,f_best,f_star='no'): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = PosteriorMean(init_X,dim,f_best,model,f_star)
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -PosteriorMean(X=x,dim=dim,f_best=f_best,model=model,f_star=f_star),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =  res.x  
    AF_temp = PosteriorMean(X=np.array(X_temp).reshape(-1,1),dim=dim,f_best=f_best,model=model,f_star=f_star)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next
