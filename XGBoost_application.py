import botorch
from known_boundary.acquisition_function import EI_acquisition_opt,MES_acquisition_opt,LCB_acquisition_opt,ERM_acquisition_opt,SLogTEI_acquisition_opt,SLogEI_acquisition_opt
from known_boundary.utlis import  get_initial_points,transform,opt_model_MLE,opt_model_MAP
import numpy as np
import GPy
import torch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,StyblinskiTang
import obj_functions.push_problems
from botorch.utils.transforms import unnormalize,normalize
from known_boundary.SLogGP import SLogGP
from obj_functions import obj_function
import scipy 

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double





    
n_init = 4*6

N = 50

step_size = 1
iter_num = 30
    
lengthscale_range = [0.001,2]
variance_range = [0.001**2,4**2]
noise = 1e-6

task_name = 'skin'

############################# GP+EI ###################################
BO_EI = []
noise = 1e-6

for exp in range(N):
    
    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
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
                
                # print('lengthscale: ',lengthscale)
                # print('variance: ',variance)
                
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
            m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            m.Gaussian_noise.fix(noise)

            np.random.seed(i)
            standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            
            print(best_record[-1])
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            print('noise: ',noise)
            
    best_record = np.array(best_record) 
    BO_EI.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_GP+EI', BO_EI, delimiter=',')


############################# GP+TEI ###################################
BO_TEI = []

for exp in range(N):
    
    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
    np.random.seed(1234)

    for i in range(iter_num):

            print(i)
        
            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)
            fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
            fstar_standard = fstar_standard.item()
            
            minimal = train_Y.min().item()
            
            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            # train the GP
            if i%step_size == 0:
                
                parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                    
                lengthscale = parameters[0]
                variance = parameters[1]
                
                # print('lengthscale: ',lengthscale)
                # print('variance: ',variance)
                
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
            m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            m.Gaussian_noise.fix(noise)
            
            np.random.seed(i)
            standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal,f_star=fstar_standard)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            
            print(best_record[-1])
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            print('noise: ',noise)
            
            
    best_record = np.array(best_record) 
    BO_TEI.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_GP+TEI', BO_TEI, delimiter=',')


##################################################### GP+MES ##################################################
BO_MES = []

for exp in range(N):

    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
    np.random.seed(1234)

    for i in range(iter_num):
        
        
            train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
            train_X = normalize(X_BO, bounds)
            
            fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
            fstar_standard = fstar_standard.item()
            
            train_Y = train_Y.numpy()
            train_X = train_X.numpy()
            
            # train the GP

            if i%step_size == 0:
            
                parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                    
                lengthscale = parameters[0]
                variance = parameters[1]

                
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
            m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            m.Gaussian_noise.fix(noise)

            np.random.seed(i)
            standard_next_X = MES_acquisition_opt(m,standard_bounds,fstar_standard)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            print(best_record[-1])
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            print('noise: ',noise)
            
    best_record = np.array(best_record) 
    BO_MES.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_GP+MES', BO_MES, delimiter=',')


###################################### ERM ##############################################
BO_ERM = []
for exp in range(N):

    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
    np.random.seed(1234)
    
    Trans = False

    for i in range(iter_num):

        print(i)
        train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
        train_X = normalize(X_BO, bounds)
        
        train_Y = train_Y.numpy()
        train_X = train_X.numpy()
        
        fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
        fstar_standard = fstar_standard.item()
        
        if not Trans:
            minimal = np.min(train_Y)
            if i%step_size == 0:
                parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                    
                lengthscale = parameters[0]
                variance = parameters[1]
                
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
            m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            m.Gaussian_noise.fix(noise)
            
            np.random.seed(i)
            standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
            
            beta = np.sqrt(np.log(train_X.shape[0]))
            _,lcb = LCB_acquisition_opt(m,standard_bounds,beta)
            if lcb < fstar_standard:
                Trans = True
                #print('transform!')
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            print('noise: ',noise)
        
        else:    
            print('trans!')                    
            train_Y_transform = transform(y=train_Y,fstar=fstar_standard)
            mean_temp = np.mean(train_Y_transform)
            
            if i%step_size == 0:
                parameters = opt_model_MLE(train_X,(train_Y_transform-mean_temp),dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range) 
                lengthscale = parameters[0]
                variance = parameters[1]
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
            m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            m.Gaussian_noise.fix(noise)
            np.random.seed(i)
            standard_next_X,erm_value = ERM_acquisition_opt(m,bounds=standard_bounds,fstar=fstar_standard,mean_temp=mean_temp)
            print(standard_next_X)
            
        
        
        if np.any(np.abs((standard_next_X - train_X)).sum(axis=1) <= (dim*3e-4)):
            print('random')
            X_next = get_initial_points(bounds, 1,device,dtype,seed=i)
        
        else:        
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)     
        
        Y_next = fun(X_next).reshape(-1,1)

        # Append data
        X_BO = torch.cat((X_BO, X_next), dim=0)
        Y_BO = torch.cat((Y_BO, Y_next), dim=0)

        best_value = float(Y_BO.min())
        best_record.append(best_value)
        print(best_record[-1])
        
        noise = variance*10**(-5)   #adaptive noise
        noise = np.round(noise, -int(np.floor(np.log10(noise))))
        print('noise: ',noise)


    best_record = np.array(best_record)
    BO_ERM.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_transformedGP+ERM', BO_ERM, delimiter=',')


######################## SlogGP+logEI#######################################
LogEI_noboundary = []

for exp in range(N):

    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
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
                # print('lower bound is ',-c)
            
            
            warp_Y = np.log(train_Y+c)
            mean_warp_Y = np.mean(warp_Y) # use to predict mean
            warp_Y_standard = warp_Y-mean_warp_Y
            
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
            m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
            m.Gaussian_noise.variance.fix(noise)
            
            np.random.seed(i)
            standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            print(best_record[-1])
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            #print('noise: ',noise)
            
            
    best_record = np.array(best_record)         
    LogEI_noboundary.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_SLogGP+logEI', LogEI_noboundary, delimiter=',')


######################## SlogGP (boundary)+logEI#######################################

LogEI_boundary = []

for exp in range(N):
    
    exp = 16

    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
    np.random.seed(1234)
    

    uncertainty_index = 1
    tolerance_level = 2.5
    
    for i in range(iter_num):

            print('inner loop: ',i)
            print('uncertainty: ',uncertainty_index)
        
            train_Y = Y_BO.numpy()
            fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound

            train_Y = train_Y - np.min(train_Y)  # shift Y

            train_X = normalize(X_BO, bounds)
            train_X = train_X.numpy()
            
            lower = -np.min(train_Y)+10**(-6)
            upper = lower+2000 
            c_range = [lower,upper]
            
            mu_prior = np.log(-fstar_shifted)  #np.log(-fstar_shifted+ 0.3)  
            sigma_prior = 0.2*uncertainty_index       #np.sqrt(np.log(-fstar_shifted+0.3)-np.log(-fstar_shifted))
            prior_parameter = [mu_prior,sigma_prior]
            

            if i%step_size == 0:
                
                if uncertainty_index<=200:
                
                    parameters = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,
                                                    prior_parameter,noise=noise,seed=i)
        
                    # lengthscale = parameters[0]
                    # variance = parameters[1]
                    c = parameters[2]
                    
                    
                    if abs(-c-fstar_shifted)> np.exp(mu_prior+tolerance_level*sigma_prior) -np.exp(mu_prior): #      100
                        temp = (np.log(abs(-c-fstar_shifted)+np.exp(mu_prior))-mu_prior)/sigma_prior - tolerance_level
                        uncertainty_index += 2*temp
                    
                        print('Not Use prior')
                        parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                    lengthscale_range=lengthscale_range,
                                                    variance_range=variance_range,c_range=c_range)                
        
                        # lengthscale = parameters[0]
                        # variance = parameters[1]
                        # c = parameters[2]
                # else: 
                #     print('Not Use prior because uncertainty is huge')
                #     parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                #                                lengthscale_range=lengthscale_range,variance_range=variance_range,
                #                                c_range=c_range)                
    
                    # lengthscale = parameters[0]
                    # variance = parameters[1]
                    # c = parameters[2]
                
                
                # print('lengthscale: ',lengthscale)
                # print('variance: ',variance)
                # print('lower bound: ',-c+np.min(train_Y))
                # print('shift fstar: ',fstar_shifted)
            
            lengthscale = parameters[0]
            variance = parameters[1]
            c = parameters[2]
            
            # print('lengthscale: ',lengthscale)
            # print('variance: ',variance)
            # print('-c: ',-c)
                        
            warp_Y = np.log(train_Y+c)
            mean_warp_Y = np.mean(warp_Y) # use to predict mean
            warp_Y_standard = warp_Y-mean_warp_Y
            
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
            m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
            m.Gaussian_noise.variance.fix(noise)
            
            np.random.seed(i)
            standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,
                                                        f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)
            

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            print('best so far: ',best_record[-1])
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            #print('noise: ',noise)
            
            
    best_record = np.array(best_record)     
    LogEI_boundary.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_SLogGP(boundary)+logEI', LogEI_boundary, delimiter=',')



######################## SlogGP (boundary)+logTEI#######################################

LogTEI_boundary = []

for exp in range(N):

    print(exp)

    seed = exp

    fun = obj_function.XGBoost(task=task_name, seed=seed)

    dim = fun.dim
    bounds = fun.bounds
    
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    fstar = 0.
    

    X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    Y_BO = torch.tensor(
        [fun(x) for x in X_BO], dtype=dtype, device=device
    ).reshape(-1,1)
    
    best_record = [Y_BO.min().item()]
    
    init_best = Y_BO.min().item()
    
    np.random.seed(1234)
    

    uncertainty_index = 1
    tolerance_level = 2.5
    
    for i in range(iter_num):

            print('inner loop: ',i)
            print('uncertainty: ',uncertainty_index)
        
            train_Y = Y_BO.numpy()
            fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound

            train_Y = train_Y - np.min(train_Y)  # shift Y

            train_X = normalize(X_BO, bounds)
            train_X = train_X.numpy()
            
            lower = -np.min(train_Y)+10**(-6)
            upper = lower+2000 
            c_range = [lower,upper]
            
            mu_prior = np.log(-fstar_shifted)  #np.log(-fstar_shifted+ 0.3)  
            sigma_prior = 0.2*uncertainty_index       #np.sqrt(np.log(-fstar_shifted+0.3)-np.log(-fstar_shifted))
            prior_parameter = [mu_prior,sigma_prior]
            

            if i%step_size == 0:
                
                if uncertainty_index<=200:
                
                    parameters = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,
                                                    prior_parameter,noise=noise,seed=i)
        
                    # lengthscale = parameters[0]
                    # variance = parameters[1]
                    c = parameters[2]
                    
                    
                    if abs(-c-fstar_shifted)> np.exp(mu_prior+tolerance_level*sigma_prior) -np.exp(mu_prior): #      100
                        temp = (np.log(abs(-c-fstar_shifted)+np.exp(mu_prior))-mu_prior)/sigma_prior - tolerance_level
                        uncertainty_index += 2*temp
                    
                        print('Not Use prior')
                        parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                    lengthscale_range=lengthscale_range,
                                                    variance_range=variance_range,c_range=c_range)                
        
                        # lengthscale = parameters[0]
                        # variance = parameters[1]
                        # c = parameters[2]
                # else: 
                #     print('Not Use prior because uncertainty is huge')
                #     parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                #                                lengthscale_range=lengthscale_range,variance_range=variance_range,
                #                                c_range=c_range)                
    
                    # lengthscale = parameters[0]
                    # variance = parameters[1]
                    # c = parameters[2]
                
                
                # print('lengthscale: ',lengthscale)
                # print('variance: ',variance)
                # print('lower bound: ',-c+np.min(train_Y))
                # print('shift fstar: ',fstar_shifted)
            
            lengthscale = parameters[0]
            variance = parameters[1]
            c = parameters[2]
            
            print('lengthscale: ',lengthscale)
            print('variance: ',variance)
            print('-c: ',-c)
                        
            warp_Y = np.log(train_Y+c)
            mean_warp_Y = np.mean(warp_Y) # use to predict mean
            warp_Y_standard = warp_Y-mean_warp_Y
            
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
            m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
            m.Gaussian_noise.variance.fix(noise)
            
            np.random.seed(i)
            if -c>=fstar_shifted:
                print('logEI')
                standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,
                                                            f_best=np.min(train_Y),
                                                            c=c,f_mean=mean_warp_Y)
            else:
                print('logTEI')
                standard_next_X = SLogTEI_acquisition_opt(model=m,bounds=standard_bounds,
                                                            f_best=np.min(train_Y),c=c,
                                                            f_mean=mean_warp_Y,fstar=fstar_shifted)  
            
            # np.random.seed(i)
            # standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,
            #                                          f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y)
            
            X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
            Y_next = fun(X_next).reshape(-1,1)
            

            # Append data
            X_BO = torch.cat((X_BO, X_next), dim=0)
            Y_BO = torch.cat((Y_BO, Y_next), dim=0)
            
            best_record.append(Y_BO.min().item())
            print('best so far: ',best_record[-1])
            
            noise = variance*10**(-5)   #adaptive noise
            noise = np.round(noise, -int(np.floor(np.log10(noise))))
            #print('noise: ',noise)
            
            
    best_record = np.array(best_record)     
    LogTEI_boundary.append(best_record)
    
np.savetxt('exp_res/'+task_name+'_SLogGP(boundary)+logTEI', LogTEI_boundary, delimiter=',')
