import botorch
from botorch.acquisition import AnalyticAcquisitionFunction

#from __future__ import annotations

import math
from abc import ABC

# from contextlib import nullcontext
# from copy import deepcopy

from typing import Dict, Optional, Tuple, Union

import torch
# from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
# from botorch.exceptions import UnsupportedError
# from botorch.models.gp_regression import FixedNoiseGP
# from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
# from botorch.utils.constants import get_constants_like
# from botorch.utils.probability import MVNXPB
from botorch.utils.probability.utils import (
    ndtr as Phi,
    phi,
)
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
# from torch.nn.functional import pad



class EI_botorch(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        mean, sigma = self._mean_and_sigma(X)

        z = (self.best_f - mean)/sigma     
        out=(self.best_f - mean) * Phi(z) + sigma * phi(z)
  
        return out
    
    
class MES_botorch(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        fstar: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        mean, sigma = self._mean_and_sigma(X)
        
        gamma = -(self.fstar-mean)/sigma

        cdf_part = Phi(gamma)
        out = (gamma*phi(gamma))/(2*cdf_part)-torch.log(cdf_part)

        return out
    
    
    
class LCB_botorch(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        mean, sigma = self._mean_and_sigma(X)
        
        out= -(mean - self.beta*sigma)
        return out
    




class ERM_botorch(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        fstar: Union[float, Tensor],
        mean_temp: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.register_buffer("mean_temp", torch.as_tensor(mean_temp))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        mean_g, sigma_g = self._mean_and_sigma(X)
        mean_g = mean_g + self.mean_temp

        mu_f = self.fstar + 1/2*mean_g**2
        sigma_f = mean_g**2 * sigma_g

        gamma = (mu_f - self.fstar)/sigma_f      
        out= - (sigma_f * phi(gamma) + (mu_f - self.fstar) * Phi(gamma))

  
        return out
    
    
    
    
class LogEI_botorch(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        c: Union[float, Tensor],
        f_mean: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("c", torch.as_tensor(c))
        self.register_buffer("f_mean", torch.as_tensor(f_mean))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        mean, sigma = self._mean_and_sigma(X)
        mu = mean+self.f_mean
        C = self.c+self.best_f
        out = C*Phi((math.log(C)-mu)/sigma)-torch.exp(mu+sigma**2/2)*Phi((math.log(C)-mu-sigma**2)/sigma)

  
        return out
    
    
class LogTEI_botorch(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        c: Union[float, Tensor],
        f_mean: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("c", torch.as_tensor(c))
        self.register_buffer("f_mean", torch.as_tensor(f_mean))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        mean, sigma = self._mean_and_sigma(X)
        
        mu = mean+self.f_mean
        C = self.c+self.best_f
        part1 = C*Phi((math.log(C)-mu)/sigma)-torch.exp(mu+sigma**2/2)*Phi((math.log(C)-mu-sigma**2)/sigma)
        
        C = self.c
        part2 = C*Phi((math.log(C)-mu)/sigma)-torch.exp(mu+sigma**2/2)*Phi((math.log(C)-mu-sigma**2)/sigma)
        
        out = part1-part2
        
        # part3 = self.best_f*Phi(  (math.log(self.c)-mu) /sigma ) 
    
        # out = out_temp+part3

  
        return out