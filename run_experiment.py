import botorch
from known_boundary.acquisition_function import EI_acquisition_opt,SLogTEI_acquisition_opt,SLogEI_acquisition_opt
from known_boundary.utlis import  get_initial_points,transform,opt_model_MLE,opt_model_MAP
import numpy as np
import GPy
import torch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann
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

temp={}
temp['name']='Hartmann3D' 
temp['function'] = Hartmann(dim=3,negate=False)
temp['fstar'] =  -3.86278
function_information.append(temp)

# temp={}
# temp['name']='Branin2D' 
# temp['function'] = Branin(negate=False)
# temp['fstar'] =  0.397887 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley2D' 
# temp['function'] = Ackley(dim=2,negate=False)
# temp['fstar'] =  0 
# function_information.append(temp)

# temp={}
# temp['name']='Beale2D' 
# temp['function'] = Beale(negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Levy2D' 
# temp['function'] = Levy(dim=2,negate=False)
# temp['fstar'] =  0.
# function_information.append(temp)

# temp={}
# temp['name']='Rosenbrock2D' 
# temp['function'] = Rosenbrock(dim=2,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='SixHumpCamel2D' 
# temp['function'] = SixHumpCamel(negate=False)
# temp['fstar'] =  -1.0317
# function_information.append(temp)




for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim
    iter_num = 50
    N = 20
    fstar = information['fstar']
    
    print('fstar is: ',fstar)
    
    if dim <=3:
        step_size = 2
    elif dim<=7:
        step_size = 3
    else:
        step_size = 4
        
    lengthscale_range = [0.001,2]
    variance_range = [0.001**2,4**2]
    noise = 1e-6
        
    
    ############################# GP+EI ###################################
    BO_EI = []
    noise = 1e-6

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
                    
                    parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    
                    noise = variance*10**(-4)
                    print('noise: ',noise)
                    
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
                
                print(best_record[-1])
                
        best_record = np.array(best_record) 
        BO_EI.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_GP+EI', BO_EI, delimiter=',')
    
    
    # ############################# GP+TEI ###################################
    # BO_TEI = []

    # for exp in range(N):
        
    #     print(exp)
        
    #     seed = exp

    #     X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    #     Y_BO = torch.tensor(
    #         [fun(x) for x in X_BO], dtype=dtype, device=device
    #     ).reshape(-1,1)

    #     best_record = [Y_BO.min().item()]
    #     np.random.seed(1234)

    #     for i in range(iter_num):

    #             print(i)
            
    #             train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
    #             train_X = normalize(X_BO, bounds)
    #             fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
    #             fstar_standard = fstar_standard.item()
                
    #             minimal = train_Y.min().item()
                
    #             train_Y = train_Y.numpy()
    #             train_X = train_X.numpy()
                
    #             # train the GP
    #             if i%step_size == 0:
                    
    #                 parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                        
    #                 lengthscale = parameters[0]
    #                 variance = parameters[1]
                    
    #                 # print('lengthscale: ',lengthscale)
    #                 # print('variance: ',variance)
                    
    #             kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
    #             m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
    #             m.Gaussian_noise.fix(noise)
                
    #             standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal,f_star=fstar_standard)
    #             X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
    #             Y_next = fun(X_next).reshape(-1,1)

    #             # Append data
    #             X_BO = torch.cat((X_BO, X_next), dim=0)
    #             Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
    #             best_record.append(Y_BO.min().item())
                
    #             print(best_record[-1])
                
    #     best_record = np.array(best_record) 
    #     BO_TEI.append(best_record)
        
    # np.savetxt('exp_res/'+information['name']+'_GP+TEI', BO_TEI, delimiter=',')
    
    
    ######################## SlogGP+logEI#######################################
    LogEI_noboundary = []

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
                upper = lower+2000 #min(300,5*train_Y_std)
                
                c_range = [lower,upper]

                if i%step_size == 0:
                    
                    parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                
        
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
                print(best_record[-1])
                
        best_record = np.array(best_record)         
        LogEI_noboundary.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_SLogGP+logEI', LogEI_noboundary, delimiter=',')
    
    
    ######################## SlogGP (boundary)+logEI#######################################
    
    LogEI_boundary = []
    
    for exp in range(N):

        seed = exp
        
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        print(best_record[-1])
        np.random.seed(1234)

        for i in range(iter_num):

                print('inner loop: ',i)
            
                train_Y = Y_BO.numpy()
                fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound
                #print('shift lower bound: ',fstar_shifted)
                train_Y = train_Y - np.min(train_Y)  # shift Y
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
        
                mu_prior = np.log(-fstar_shifted+0.3)
                sigma_prior = np.sqrt(np.log(-fstar_shifted+0.3)-np.log(-fstar_shifted))
                prior_parameter = [mu_prior,sigma_prior]
                
                # sigma = 0.3
                # prior_parameter = [np.log(-fstar_shifted)+sigma**2,sigma]
                
                if i%step_size == 0:
                    
                    best_parameter = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,prior_parameter,noise=noise,seed=i)
                    

        
                    lengthscale = best_parameter[0]
                    variance = best_parameter[1]
                    c = best_parameter[2]
                    
                    # print('lengthscale: ',lengthscale)
                    # print('variance: ',variance)
                    # print('c: ',c)
                
                
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
                print('best so far: ',best_record[-1])
                
        best_record = np.array(best_record)     
        LogEI_boundary.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_SLogGP(boundary)+logEI', LogEI_boundary, delimiter=',')
    



    ######################## SlogGP (boundary)+logTEI#######################################
    
    LogTEI_boundary = []
    
    for exp in range(N):

        seed = exp
        
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        print(best_record[-1])
        np.random.seed(1234)

        for i in range(iter_num):

                print('inner loop: ',i)
            
                train_Y = Y_BO.numpy()
                fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound
                train_Y = train_Y - np.min(train_Y)  # shift Y
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
        
                mu_prior = np.log(-fstar_shifted+0.3)
                sigma_prior = np.sqrt(np.log(-fstar_shifted+0.3)-np.log(-fstar_shifted))
                prior_parameter = [mu_prior,sigma_prior]
                
                # sigma = 0.3
                # prior_parameter = [np.log(-fstar_shifted)+sigma**2,sigma]
                
                if i%step_size == 0:
                    
                    best_parameter = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,prior_parameter,noise=noise,seed=i)
                    
                    lengthscale = best_parameter[0]
                    variance = best_parameter[1]
                    c = best_parameter[2]
                    
                    # print('lengthscale: ',lengthscale)
                    # print('variance: ',variance)
                    # print('c: ',c)
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(noise)
                
                if -c>fstar_shifted:
                    #print('logEI')
                    standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
                else:
                    #print('logTEI')
                    standard_next_X = SLogTEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y,fstar=fstar_shifted)
                
                X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
                Y_next = fun(X_next).reshape(-1,1)
                
                
                

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                print('best so far: ',best_record[-1])
                
        best_record = np.array(best_record)     
        LogTEI_boundary.append(best_record)
        
    np.savetxt('exp_res/'+information['name']+'_SLogGP(boundary)+logTEI', LogTEI_boundary, delimiter=',')
    
    