import torch
from botorch.utils.sampling import draw_sobol_samples
import numpy as np
import GPy
from known_boundary.SLogGP import SLogGP

import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def get_initial_points(bounds,num,device,dtype,seed=0):
    
        train_x = draw_sobol_samples(
        bounds=bounds, n=num, q=1,seed=seed).reshape(num,-1).to(device, dtype=dtype)
        
        return train_x
    
    
def transform(y,fstar):
  y_transformed = np.sqrt(2*(y-fstar))
  return y_transformed



def range_transform(lower1,upper1,lower2,upper2,ratio): #lower 2 must be larger than lower 1
    
    if upper1>upper2:
        upper_dist = upper1-upper2
        upper_res = upper2+ratio*upper_dist
    else:
        upper_res = upper2
        
    lower_dist = lower2-lower1
    lower_res = lower2 - ratio*lower_dist
    
    return lower_res,upper_res


def opt_model(train_X,train_Y,dim,model_type,noise=1e-5,**kwargs):
    
    obj_holder = []
    parameter_holder = []
    
    
    if model_type == 'GP':
        
        lengthscale_range = kwargs['lengthscale_range']
        variance_range = kwargs['variance_range']
        
        parameter_num = 2
        
        for ii in range(int(3**parameter_num/2)+1):
            
            np.random.seed(ii)
            
            lengthscale_init = np.random.uniform(lengthscale_range[0],lengthscale_range[1],1)[0]
            variance_init = np.random.uniform(variance_range[0],variance_range[1],1)[0]
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale_init,variance=variance_init)
            m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            m.rbf.lengthscale.constrain_bounded(lengthscale_range[0],lengthscale_range[1])
            m.rbf.variance.constrain_bounded(variance_range[0],variance_range[1])
            m.Gaussian_noise.fix(noise)
            
            m.optimize()
            
            obj_temp = -m.log_likelihood()
            lengthscale_temp = m.rbf.lengthscale.item()
            variance_temp = m.rbf.variance.item()
            
            obj_holder.append(obj_temp)
            parameter_holder.append([lengthscale_temp,variance_temp])
            
    elif model_type == 'SLogGP':
        
        lengthscale_range = kwargs['lengthscale_range']
        variance_range = kwargs['variance_range']
        c_range = kwargs['c_range']
                
        parameter_num = 3
        
        for ii in range(int(3**parameter_num/2)+1):
            
            np.random.seed(ii)
            
            lengthscale_init = np.random.uniform(lengthscale_range[0],lengthscale_range[1],1)[0]
            variance_init = np.random.uniform(variance_range[0],variance_range[1],1)[0]
            c_init = np.random.uniform(c_range[0],c_range[1],1)[0]
            
            m = GPy.models.WarpedGP(train_X.reshape(-1,dim), train_Y.reshape(-1,1),warping_function=SLogGP(lower=c_range[0],upper=c_range[1],n_terms =1))
            m.rbf.lengthscale = lengthscale_init
            m.rbf.variance = variance_init
            m.SlogGP.psi = c_init
            
            m.rbf.lengthscale.constrain_bounded(lengthscale_range[0],lengthscale_range[1])
            m.rbf.variance.constrain_bounded(variance_range[0],variance_range[1])
            
            m.optimize()
            
            obj_temp = -m.log_likelihood()
            lengthscale_temp = m.rbf.lengthscale.item()
            variance_temp = m.rbf.variance.item()
            c_temp = m.SlogGP.psi.item()
            
            obj_holder.append(obj_temp)
            parameter_holder.append([lengthscale_temp,variance_temp,c_temp])
            
    index = np.argmin(obj_holder)
    
    return parameter_holder[index]
            
    