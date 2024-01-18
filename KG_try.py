import botorch
from known_boundary.utlis import  get_initial_points,transform,opt_model_MLE,opt_model_MAP
from known_boundary.Knowledge_Gradient import  DiscreteKnowledgeGradient,HybridKnowledgeGradient
from botorch.acquisition import PosteriorMean #ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import numpy as np
import GPy
import torch
from botorch.test_functions import Rastrigin,Ackley,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,DixonPrice,Levy,StyblinskiTang,Griewank
import obj_functions.push_problems
from botorch.utils.transforms import unnormalize,normalize
from known_boundary.SLogGP import SLogGP
import scipy 
from botorch.utils.sampling import manual_seed
from known_boundary.acquisition_function_Botorch import EI_botorch, LogEI_botorch

from botorch.models import SingleTaskGP,FixedNoiseGP
from botorch.acquisition import ExpectedImprovement,PosteriorMean,qKnowledgeGradient
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine
from gpytorch.kernels import MaternKernel, RBFKernel, IndexKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means import ZeroMean

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []


temp={}
temp['name']='Push4D' 
f_class = obj_functions.push_problems.push4
tx_1 = 3.5; ty_1 = 4
fun = f_class(tx_1, ty_1,negate=True)
temp['function'] = fun
temp['fstar'] =  0.
function_information.append(temp)

# temp={}
# temp['name']='Branin2D' 
# temp['function'] = Branin(negate=True)
# temp['fstar'] =  -0.397887 
# function_information.append(temp)

# temp={}
# temp['name']='Beale2D' 
# temp['function'] = Beale(negate=True)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley6D' 
# temp['function'] = Ackley(dim=6,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)


# temp={}
# temp['name']='SixHumpCamel2D' 
# temp['function'] = SixHumpCamel(negate=True)
# temp['fstar'] =  -1.0317
# function_information.append(temp)


# temp={}
# temp['name']='Hartmann3D' 
# temp['function'] = Hartmann(dim=3,negate=True)
# temp['fstar'] =  -3.86278
# function_information.append(temp)


# temp={}
# temp['name']='DixonPrice4D' 
# temp['function'] = DixonPrice(dim=4,negate=True)
# temp['fstar'] = 0.
# temp['min']=True 
# function_information.append(temp)

# temp={}
# temp['name']='Rosenbrock5D' 
# temp['function'] = Rosenbrock(dim=5,negate=True)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley6D' 
# temp['function'] = Ackley(dim=6,negate=True)
# temp['fstar'] =  0. 
# function_information.append(temp)


# temp={}
# temp['name']='Powell8D' 
# temp['function'] = Powell(dim=8,negate=True)
# temp['fstar'] = 0. 
# temp['min']=True 
# function_information.append(temp)


# temp={}
# temp['name']='StyblinskiTang10D' 
# temp['function'] = StyblinskiTang(dim=10,bounds = [(-3., 3.) for _ in range(10)],negate=False)
# temp['fstar'] = -10*39.166166
# temp['min']=True 
# function_information.append(temp)


# temp={}
# temp['name']='Rastrigin10D' 
# temp['function'] = Rastrigin(dim=10,negate=False)
# temp['fstar'] = 0.
# temp['min']=True 
# function_information.append(temp)



# temp={}
# temp['name']='Levy10D' 
# temp['function'] = Levy(dim=10,negate=False)
# temp['fstar'] = 0. 
# temp['min']=True 
# function_information.append(temp)

for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim

    
    fstar = information['fstar']
    
    print('fstar is: ',fstar)
    
    if dim <=3:
        step_size = 2
        iter_num = 50
        N = 100
    elif dim<=5:
        step_size = 3
        iter_num = 100
        N = 50
    else:
        step_size = 3
        iter_num = 150
        N = 3
        
    lengthscale_range = [0.001,2]
    variance_range = [0.001**2,20]
    noise = 1e-6
    
    print(information['name'])

    
    ############################# GP+KG ###################################
    BO_KG = []
    noise = 1e-6

    for exp in range(N):
        
        print(exp)
        
        seed = exp

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.max().item()]
        np.random.seed(1234)

        for i in range(iter_num):

                print(i)
                
                if i%step_size == 0:
                    Y_mean =  Y_BO.mean()
                    Y_std = Y_BO.std()
            
                train_Y = (Y_BO -Y_mean) / Y_std
                train_X = normalize(X_BO, bounds)
                
                
                train_Y = train_Y.numpy()
                train_X = train_X.numpy()
                
                # train the GP
                if i%step_size == 0:
                    
                    parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    
                
                covar_module =  ScaleKernel(RBFKernel())
                train_yvar = torch.tensor(noise, device=device, dtype=dtype)

                torch.manual_seed(exp+iter_num)
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(train_Y), train_yvar.expand_as(torch.tensor(train_Y)),
                                     mean_module=ZeroMean(),covar_module=covar_module).to(device)
                
                model.covar_module.outputscale = variance
                model.covar_module.base_kernel.lengthscale = lengthscale
                
                model.eval()
                
                AF = ExpectedImprovement(model=model, best_f=np.max(train_Y)) .to(device)
                standard_next_X_final, _ = optimize_acqf(
                    acq_function=AF,
                    bounds=torch.tensor(standard_bounds.T) .to(device),
                    q=1,
                    num_restarts=3*dim,
                    raw_samples=30*dim,
                    options={},
                )

                
                X_next_final = unnormalize(standard_next_X_final, bounds).reshape(-1,dim)            
                Y_next_final = fun(X_next_final).reshape(-1,1)

                # Append data
                X_BO_final = torch.cat((X_BO, X_next_final), dim=0)
                Y_BO_final = torch.cat((Y_BO, Y_next_final), dim=0)
                
                best_record.append(Y_BO_final.max().item())
                
                print(best_record[-1])
                
                AF = DiscreteKnowledgeGradient(model=model, bounds=torch.tensor(standard_bounds.T),num_discrete_points=300) .to(device)
                
                # AF = HybridKnowledgeGradient(
                #                 model=model,
                #                 bounds=torch.tensor(standard_bounds.T),
                #                 num_fantasies=5,
                #                 num_restarts=3*dim,
                #                 raw_samples=30*dim,
                #             )
                
                
                #AF = qKnowledgeGradient(model, num_fantasies=16)
                
                
                #with manual_seed(1234):
                standard_next_X, _ = optimize_acqf(
                    acq_function=AF,
                    bounds=torch.tensor(standard_bounds.T) .to(device),
                    q=1,
                    num_restarts=3*dim,
                    raw_samples=30*dim,
                    options={},
                )

                print('KG pick: ',standard_next_X)
                
                # check whether it is too close to the historical data
                X_next = unnormalize(standard_next_X, bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                noise = variance*10**(-5)   #adaptive noise
                noise = np.round(noise, -int(np.floor(np.log10(noise))))
                print('noise: ',noise)
                
        best_record = np.array(best_record) 
        BO_KG.append(best_record)
            
    np.savetxt('KG/'+information['name']+'_GP+KG_part1', BO_KG, delimiter=',')
    