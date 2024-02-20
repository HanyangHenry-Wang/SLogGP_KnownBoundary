import numpy as np
from GPy.core.parameterization import Parameterized, Param
from paramz.transformations import Logexp
from GPy.util.warping_functions import WarpingFunction
import sys


class SLogGP(WarpingFunction):
    """
    This is the function proposed in Snelson et al.:
    A sum of tanh functions with linear trends outside
    the range. Notice the term 'd', which scales the
    linear trend.
    """
    def __init__(self, lower,upper,n_terms =1, psi_bound = True, initial_y=None):
        """
        n_terms specifies the number of tanh terms to be used
        """
        self.n_terms = n_terms
        self.num_parameters = 1
        self.psi = np.ones((self.n_terms, 1))
        super(SLogGP, self).__init__(name='SlogGP')
        self.psi = Param('psi', self.psi)  #Param will register psi as a parameter
        if psi_bound:
            self.psi[:, :].constrain_bounded(lower,upper) #put constraint in the psi space
        else:
            self.psi[:, :].constrain_positive()
            
        self.link_parameter(self.psi)
        self.initial_y = initial_y

    def f(self, y):  # f is the warping function

        psi = self.psi
        
        z_temp = np.log(y+psi)
        mean_temp = np.mean(z_temp)
        
        z = z_temp-mean_temp

        return z

    def fgrad_y(self, y, return_precalc=False):
        """
        gradient of f w.r.t to y ([N x 1])

        :returns: Nx1 vector of derivatives, unless return_precalc is true, 
        then it also returns the precomputed stuff
        """

        psi = self.psi
        
        N = y.shape[0]

        GRAD = (N-1)/N *1/(y+psi)

        return GRAD

    def fgrad_y_psi(self, y, return_covar_chain=False):
        """
        gradient of f w.r.t to y and psi

        :returns: NxIx4 tensor of partial derivatives
        """
        psi = self.psi
        
        N = y.shape[0]

        gradients = np.zeros((y.shape[0], y.shape[1], len(psi), 1))
        for i in range(len(psi)):
    
            gradients[:, :, i, 0] = (N-1)/N * (-1/(y+psi)**2)  #this should be of the shape Nx1
            

        if return_covar_chain:
            covar_grad_chain = np.zeros((y.shape[0], y.shape[1], len(psi), 1))
            
            
            for i in range(len(psi)):
                mean_temp = np.mean(1/(y+psi))
                covar_grad_chain[:, :, i, 0] = 1/(y+psi)-mean_temp
                
            return gradients, covar_grad_chain

        return gradients

    def update_grads(self, Y_untransformed, Kiy):
        grad_y = self.fgrad_y(Y_untransformed)
        grad_y_psi, grad_psi = self.fgrad_y_psi(Y_untransformed,
                                                return_covar_chain=True)
        djac_dpsi = ((1.0 / grad_y[:, :, None, None]) * grad_y_psi).sum(axis=0).sum(axis=0)
        dquad_dpsi = (Kiy[:, None, None, None] * grad_psi).sum(axis=0).sum(axis=0)

        warping_grads = -dquad_dpsi + djac_dpsi

        self.psi.gradient[:] = warping_grads[:, :]