import botorch
from known_boundary.acquisition_function import EI_acquisition_opt,SLogTEI_acquisition_opt,SLogEI_acquisition_opt
from known_boundary.utlis import  get_initial_points,transform,opt_model
import numpy as np
import GPy
import torch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Rosenbrock,SixHumpCamel
from botorch.utils.transforms import unnormalize,normalize
from known_boundary.SLogGP import SLogGP
import scipy 

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []


# temp={}
# temp['name']='Branin2D' 
# temp['function'] = Branin(negate=False)
# temp['fstar'] =  0.397887 
# function_information.append(temp)

temp={}
temp['name']='Ackley2D' 
temp['function'] = Ackley(dim=2,negate=False)
temp['fstar'] =  0 
function_information.append(temp)

temp={}
temp['name']='Beale2D' 
temp['function'] = Beale(negate=False)
temp['fstar'] =  0. 
function_information.append(temp)

temp={}
temp['name']='Levy2D' 
temp['function'] = Levy(dim=2,negate=False)
temp['fstar'] =  0.
function_information.append(temp)

temp={}
temp['name']='Rosenbrock2D' 
temp['function'] = Rosenbrock(dim=2,negate=False)
temp['fstar'] =  0. 
function_information.append(temp)

temp={}
temp['name']='SixHumpCamel2D' 
temp['function'] = SixHumpCamel(negate=False)
temp['fstar'] =  -1.0317
function_information.append(temp)




for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim
    iter_num = 50
    N = 25
    fstar = information['fstar']
    
    print('fstar is: ',fstar)
    
    if dim <=3:
        step_size = 2
    elif dim<=7:
        step_size = 3
    else:
        step_size = 4
        
    lengthscale_range = [0.01,2]
    variance_range = [0.01**2,4**2]
    noise = 1e-5
        
    
    ############################# GP+EI ###################################
    BO_EI = []

    for exp in range(N):
        
        print(exp)
        
        seed = exp

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.min().item()]
        np.random.seed(1234)

        for i in range(iter_num):

                print(i)
            
                train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
                train_X = normalize(X_BO, bounds)
                
                minimal = train_Y.min().item()
                
                train_Y = train_Y.numpy()
                train_X = train_X.numpy()
                
                # train the GP
                if i%step_size == 0:
                    
                    parameters = opt_model(train_X,train_Y,dim,'GP',noise=noise,lengthscale_range=lengthscale_range,variance_range=variance_range)
                        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    
                    # print('lengthscale: ',lengthscale)
                    # print('variance: ',variance)
                    
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
                m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
                m.Gaussian_noise.fix(noise)
                
                standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                
                #print(best_record[-1])
                
        best_record = np.array(best_record) 
        BO_EI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_GP+EI', BO_EI, delimiter=',')
    
    
    ######################## SlogGP+logEI#######################################
    SLogEI_noboundary = []

    for exp in range(N):

        seed = exp
        
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        np.random.seed(1234)

        for i in range(iter_num):

                print('inner loop: ',i)
            
                train_Y = Y_BO.numpy()
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                train_Y_std = np.std(train_Y)
                lower = -np.min(train_Y)+10**(-6)
                upper = lower+min(300,5*train_Y_std)
                
                c_range = [lower,upper]

                if i%step_size == 0:
                    
                    parameters = opt_model(train_X,train_Y,dim,'SLogGP',noise=noise,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                
        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    c = parameters[2]
                    
                    # print('lengthscale is ',lengthscale)
                    # print('variance is ',variance)
                    # print('c is ',c)
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(noise)
                
                standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                #print(best_record[-1])
                
        best_record = np.array(best_record)         
        SLogEI_noboundary.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_SLogGP+logEI', SLogEI_noboundary, delimiter=',')
    
    
    ########################## enforced boundary ##################################
    
    SLogEI_enforceboundary = []

    for exp in range(N):

        seed = exp
        
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        np.random.seed(1234)

        for i in range(iter_num):

                #print('inner loop: ',i)
            
                train_Y = Y_BO.numpy()
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                lower = -fstar+10**(-6)
                upper = lower+0.3
                
                c_range = [lower,upper]

                if i%step_size == 0:
                    
                    parameters = opt_model(train_X,train_Y,dim,'SLogGP',noise=noise,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                
        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    c = parameters[2]
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(noise)
                
                standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                #print(best_record[-1])
                
        best_record = np.array(best_record)         
        SLogEI_enforceboundary.append(best_record)

    np.savetxt('exp_res/'+information['name']+'_SLogGP+logEI(enforceboundary)', SLogEI_enforceboundary, delimiter=',')
    
    ########################## enforced boundary ##################################
    
    SLogTEI_enforceboundary = []

    for exp in range(N):

        seed = exp
        
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        np.random.seed(1234)

        for i in range(iter_num):

                #print('inner loop: ',i)
            
                train_Y = Y_BO.numpy()
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                lower = -fstar+10**(-6)
                upper = lower+0.3
                
                c_range = [lower,upper]

                if i%step_size == 0:
                    
                    parameters = opt_model(train_X,train_Y,dim,'SLogGP',noise=noise,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                
        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    c = parameters[2]
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(noise)
                
                standard_next_X = SLogTEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y,fstar=fstar)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                #print(best_record[-1])
                
        best_record = np.array(best_record)         
        SLogTEI_enforceboundary.append(best_record)

    np.savetxt('exp_res/'+information['name']+'_SLogGP+logTEI(enforceboundary)', SLogTEI_enforceboundary, delimiter=',')
    
    
    ######################### no boundary + logTEI ##################################
    
    SLogTEI_enforceboundary = []

    for exp in range(N):

        seed = exp
        
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        np.random.seed(1234)

        for i in range(iter_num):

                #print('inner loop: ',i)
            
                train_Y = Y_BO.numpy()
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                train_Y_std = np.std(train_Y)
                lower = -np.min(train_Y)+10**(-6)
                upper = lower+min(300,5*train_Y_std)
                
                c_range = [lower,upper]

                if i%step_size == 0:
                    
                    parameters = opt_model(train_X,train_Y,dim,'SLogGP',noise=noise,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                
        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    c = parameters[2]
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(noise)
                
                if -c>fstar: # we do not truncate
                    #print('logEI!!')
                    standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
                else:
                    #print('logTEI!!')
                    standard_next_X = SLogTEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y,fstar=fstar)
                #print(standard_next_X)
            
                #standard_next_X = SLogTEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y,fstar=fstar)
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                #print(best_record[-1])
                
        best_record = np.array(best_record)         
        SLogTEI_enforceboundary.append(best_record)

    np.savetxt('exp_res/'+information['name']+'_SLogGP+logTEI', SLogTEI_enforceboundary, delimiter=',')