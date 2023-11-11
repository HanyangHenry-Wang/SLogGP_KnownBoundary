import botorch
from known_boundary.acquisition_function import EI_acquisition_opt,MES_acquisition_opt,LCB_acquisition_opt,ERM_acquisition_opt,SLogTEI_acquisition_opt,SLogEI_acquisition_opt
from known_boundary.utlis import  get_initial_points,transform,opt_model_MLE,opt_model_MAP
import numpy as np
import GPy
import torch
from botorch.test_functions import Ackley,Levy,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,StyblinskiTang,DixonPrice,Michalewicz
import obj_functions.push_problems
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
# temp['name']='Push4D' 
# f_class = obj_functions.push_problems.push4
# tx_1 = 3.5; ty_1 = 4
# fun = f_class(tx_1, ty_1)
# temp['function'] = fun
# temp['fstar'] =  0.
# function_information.append(temp)

temp={}
temp['name']='Branin2D' 
temp['function'] = Branin(negate=False)
temp['fstar'] =  0.397887 
function_information.append(temp)


temp={}
temp['name']='Beale2D' 
temp['function'] = Beale(negate=False)
temp['fstar'] =  0. 
function_information.append(temp)

temp={}
temp['name']='SixHumpCamel2D' 
temp['function'] = SixHumpCamel(negate=False)
temp['fstar'] =  -1.0317
function_information.append(temp)

# temp={}
# temp['name']='Hartmann3D' 
# temp['function'] = Hartmann(dim=3,negate=False)
# temp['fstar'] =  -3.86278
# function_information.append(temp)

# temp={}
# temp['name']='Hartmann6D' 
# temp['function'] = Hartmann(dim=6,negate=False)
# temp['fstar'] =  -3.32237
# function_information.append(temp)

# temp={}
# temp['name']='DixonPrice2D' 
# temp['function'] = DixonPrice(dim=2,negate=False)
# temp['fstar'] = 0.
# temp['min']=True 
# function_information.append(temp)

# temp={}
# temp['name']='Rosenbrock2D' 
# temp['function'] = Rosenbrock(dim=2,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley2D' 
# temp['function'] = Ackley(dim=2,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)


# temp={}
# temp['name']='Powell8D' 
# temp['function'] = Powell(dim=8,negate=False)
# temp['fstar'] = 0. 
# temp['min']=True 
# function_information.append(temp)



for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    n_init = 4*dim

    N = 100
    fstar = information['fstar']
    
    print('fstar is: ',fstar)
    
    if dim <=3:
        step_size = 2
        iter_num = 50
    elif dim<=8:
        step_size = 2
        iter_num = 80
    else:
        step_size = 3
        iter_num = 150
        
    lengthscale_range = [0.001,2]
    variance_range = [0.001**2,4**2]
    noise = 1e-6
    
    print(information['name'])
        
    
    # ############################# GP+EI ###################################
    # BO_EI = []
    # noise = 1e-6

    # for exp in range(N):
        
    #     print(exp)
        
    #     seed = exp+50

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

    #             np.random.seed(i)
    #             standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
    #             X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
    #             Y_next = fun(X_next).reshape(-1,1)

    #             # Append data
    #             X_BO = torch.cat((X_BO, X_next), dim=0)
    #             Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
    #             best_record.append(Y_BO.min().item())
                
    #             print(best_record[-1])
                
    #             noise = variance*10**(-5)   #adaptive noise
    #             noise = np.round(noise, -int(np.floor(np.log10(noise))))
    #             print('noise: ',noise)
                
    #     best_record = np.array(best_record) 
    #     BO_EI.append(best_record)
        
    # np.savetxt('exp_res/low_dimension/'+information['name']+'_GP+EI2', BO_EI, delimiter=',')
    
    
    # ############################# GP+TEI ###################################
    # BO_TEI = []

    # for exp in range(N):
        
    #     print(exp)
        
    #     seed = exp+50

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
                
    #             np.random.seed(i)
    #             standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal,f_star=fstar_standard)
    #             X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
    #             Y_next = fun(X_next).reshape(-1,1)

    #             # Append data
    #             X_BO = torch.cat((X_BO, X_next), dim=0)
    #             Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
    #             best_record.append(Y_BO.min().item())
                
    #             print(best_record[-1])
                
    #             noise = variance*10**(-5)   #adaptive noise
    #             noise = np.round(noise, -int(np.floor(np.log10(noise))))
    #             print('noise: ',noise)
                
                
    #     best_record = np.array(best_record) 
    #     BO_TEI.append(best_record)
        
    # np.savetxt('exp_res/low_dimension'+information['name']+'_GP+TEI2', BO_TEI, delimiter=',')
    
    
#     ##################################################### GP+MES ##################################################
#     BO_MES = []

#     for exp in range(N):

#         seed = exp
        
#         print(exp)
    
#         X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
#         Y_BO = torch.tensor(
#             [fun(x) for x in X_BO], dtype=dtype, device=device
#         ).reshape(-1,1)
        

#         best_record = [Y_BO.min().item()]

#         np.random.seed(1234)

#         for i in range(iter_num):
            
            
#                 train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
#                 train_X = normalize(X_BO, bounds)
                
#                 fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
#                 fstar_standard = fstar_standard.item()
                
#                 train_Y = train_Y.numpy()
#                 train_X = train_X.numpy()
                
#                 # train the GP
   
#                 if i%step_size == 0:
                
#                     parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                        
#                     lengthscale = parameters[0]
#                     variance = parameters[1]

                    
#                 kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
#                 m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
#                 m.Gaussian_noise.fix(noise)

#                 np.random.seed(i)
#                 standard_next_X = MES_acquisition_opt(m,standard_bounds,fstar_standard)
#                 X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)            
#                 Y_next = fun(X_next).reshape(-1,1)

#                 # Append data
#                 X_BO = torch.cat((X_BO, X_next), dim=0)
#                 Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
#                 best_record.append(Y_BO.min().item())
#                 print(best_record[-1])
                
#                 noise = variance*10**(-5)   #adaptive noise
#                 noise = np.round(noise, -int(np.floor(np.log10(noise))))
#                 print('noise: ',noise)
                
#         best_record = np.array(best_record) 
#         BO_MES.append(best_record)
        
#     np.savetxt('exp_res/'+information['name']+'_GP+MES', BO_MES, delimiter=',')
    
    
#   ###################################### ERM ##############################################
#     BO_ERM = []
#     for exp in range(N):

#         print(exp)  
#         seed = exp
        
#         Trans = False

#         X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
#         Y_BO = torch.tensor(
#                     [fun(x) for x in X_BO], dtype=dtype, device=device
#                 ).reshape(-1,1)

#         best_record = [Y_BO.min().item()]

#         np.random.seed(1234)

#         for i in range(iter_num):

#             print(i)
#             train_Y = (Y_BO - Y_BO.mean()) / Y_BO.std()
#             train_X = normalize(X_BO, bounds)
            
#             train_Y = train_Y.numpy()
#             train_X = train_X.numpy()
            
#             fstar_standard = (fstar - Y_BO.mean()) / Y_BO.std()
#             fstar_standard = fstar_standard.item()
            
#             if not Trans:
#                 minimal = np.min(train_Y)
#                 if i%step_size == 0:
#                     parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                        
#                     lengthscale = parameters[0]
#                     variance = parameters[1]
                    
#                 kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
#                 m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
#                 m.Gaussian_noise.fix(noise)
                
#                 np.random.seed(i)
#                 standard_next_X = EI_acquisition_opt(m,bounds=standard_bounds,f_best=minimal)
                
#                 beta = np.sqrt(np.log(train_X.shape[0]))
#                 _,lcb = LCB_acquisition_opt(m,standard_bounds,beta)
#                 if lcb < fstar_standard:
#                     Trans = True
#                     #print('transform!')
                
#                 noise = variance*10**(-5)   #adaptive noise
#                 noise = np.round(noise, -int(np.floor(np.log10(noise))))
#                 print('noise: ',noise)
            
#             else:    
#                 print('trans!')                    
#                 train_Y_transform = transform(y=train_Y,fstar=fstar_standard)
#                 mean_temp = np.mean(train_Y_transform)
                
#                 if i%step_size == 0:
#                     parameters = opt_model_MLE(train_X,(train_Y_transform-mean_temp),dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range) 
#                     lengthscale = parameters[0]
#                     variance = parameters[1]
                
#                 kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
#                 m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
#                 m.Gaussian_noise.fix(noise)
#                 np.random.seed(i)
#                 standard_next_X,erm_value = ERM_acquisition_opt(m,bounds=standard_bounds,fstar=fstar_standard,mean_temp=mean_temp)
#                 print(standard_next_X)
                
            
            
#             if np.any(np.abs((standard_next_X - train_X)).sum(axis=1) <= (dim*3e-4)):
#                 print('random')
#                 X_next = get_initial_points(bounds, 1,device,dtype,seed=i)
            
#             else:        
#                 X_next = unnormalize(torch.tensor(standard_next_X), bounds).reshape(-1,dim)     
            
#             Y_next = fun(X_next).reshape(-1,1)

#             # Append data
#             X_BO = torch.cat((X_BO, X_next), dim=0)
#             Y_BO = torch.cat((Y_BO, Y_next), dim=0)

#             best_value = float(Y_BO.min())
#             best_record.append(best_value)
#             print(best_record[-1])
            
#             noise = variance*10**(-5)   #adaptive noise
#             noise = np.round(noise, -int(np.floor(np.log10(noise))))
#             print('noise: ',noise)


#         best_record = np.array(best_record)
#         BO_ERM.append(best_record)
        
#     np.savetxt('exp_res/'+information['name']+'_transformedGP+ERM', BO_ERM, delimiter=',')
    
    
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
                fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound

                Y_min = np.min(train_Y)
                train_Y = train_Y - Y_min  # shift Y
                
                #scalise Y_shift and fstar_shift
                Y_std = np.std(train_Y)
                train_Y = train_Y/Y_std
                fstar_shifted = fstar_shifted/Y_std
                
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                lower = -np.min(train_Y)+10**(-6)
                upper = -fstar_shifted+20
                
                c_range = [lower,upper]

                if i%step_size == 0:
                    
                    parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                
        
                    lengthscale = parameters[0]
                    variance = parameters[1]
                    c = parameters[2]
                    
                    print('lengthscale is ',lengthscale)
                    print('variance is ',variance)
                    print('lower bound is ',-c*Y_std+Y_min)
                
                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y
                
                
                kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
                m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
                m.Gaussian_noise.variance.fix(noise)
                
                np.random.seed(i)
                standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,
                                                         f_mean=mean_warp_Y)
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
        
    np.savetxt('exp_res/low_dimension/'+information['name']+'_SLogGP+logEI', LogEI_noboundary, delimiter=',')
    
    
    
 ######################## SlogGP (boundary)+logEI#######################################
    
    LogEI_boundary = []
    sigma_holder = []
    
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
        
        tolerance_level = 2.5

        
        uncertainty = 1

        
        for i in range(iter_num):

                print('inner loop: ',i)
                print('uncertainty: ',uncertainty)
                # print('sigma prior: ',sigma_prior)
            
                train_Y = Y_BO.numpy()
                fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound
                
                Y_min = np.min(train_Y)
                train_Y = train_Y - Y_min  # shift Y
                
                #scalise Y_shift and fstar_shift
                Y_std = np.std(train_Y)
                train_Y = train_Y/Y_std
                fstar_shifted = fstar_shifted/Y_std

                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                lower = -np.min(train_Y)+10**(-6)
                upper = -fstar_shifted+20
                c_range = [lower,upper]
                
                
                
                mu_prior = np.log(-fstar_shifted) 
                sigma_prior = np.sqrt(2*(np.log(-fstar_shifted+0.1/Y_std)-mu_prior)) * uncertainty 
                print('sigma prior: ',sigma_prior)
                  
                prior_parameter = [mu_prior,sigma_prior]
                

                if i%step_size == 0:
                    
                    if sigma_prior<5:
                                    
                        parameters = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,c_range,
                                                        prior_parameter,noise=noise,seed=i)

                        c = parameters[2]
                        
                        temp = (abs(np.log(c) - mu_prior))/sigma_prior
                        #print('temp: ',temp)
                        
                        if abs(np.log(c) - mu_prior)>tolerance_level*sigma_prior :
                            temp = (abs(np.log(c) - mu_prior))/ np.sqrt(2*(np.log(-fstar_shifted+0.1)-mu_prior))
                            uncertainty = temp
                        
                            print('Not Use prior')
                
                            parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                        lengthscale_range=lengthscale_range,
                                                        variance_range=variance_range,c_range=c_range)  
                        
                    else:
                        print('sigma is big!!')
                        
                        parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                            lengthscale_range=lengthscale_range,
                                                            variance_range=variance_range,c_range=c_range) 
                            
            

                
                lengthscale = parameters[0]
                variance = parameters[1]
                c = parameters[2]
                
                # print('lengthscale: ',lengthscale)
                # print('variance: ',variance)
                print('lower bound is ',-c*Y_std+Y_min)
                            
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
        

        
    np.savetxt('exp_res/low_dimension/'+information['name']+'_SLogGP(boundary)+logEI', LogEI_boundary, delimiter=',')

 ######################## SlogGP (boundary)+logTEI#######################################
    
    LogTEI_boundary = []
    sigma_holder = []
    
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
        
        tolerance_level = 2.5

        
        uncertainty = 1

        
        for i in range(iter_num):

                print('inner loop: ',i)
                print('uncertainty: ',uncertainty)
                # print('sigma prior: ',sigma_prior)
            
                train_Y = Y_BO.numpy()
                fstar_shifted = fstar - np.min(train_Y)  # shifted lower bound
                
                Y_min = np.min(train_Y)
                train_Y = train_Y - Y_min  # shift Y
                
                #scalise Y_shift and fstar_shift
                Y_std = np.std(train_Y)
                train_Y = train_Y/Y_std
                fstar_shifted = fstar_shifted/Y_std

                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
                
                lower = -np.min(train_Y)+10**(-6)
                upper = -fstar_shifted+20
                c_range = [lower,upper]
                
                
                
                mu_prior = np.log(-fstar_shifted) 
                sigma_prior = np.sqrt(2*(np.log(-fstar_shifted+0.1/Y_std)-mu_prior)) * uncertainty 
                print('sigma prior: ',sigma_prior)
                  
                prior_parameter = [mu_prior,sigma_prior]
                

                if i%step_size == 0:
                    
                    if sigma_prior<5:
                                    
                        parameters = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,c_range,
                                                        prior_parameter,noise=noise,seed=i)

                        c = parameters[2]
                        
                        temp = (abs(np.log(c) - mu_prior))/sigma_prior
                        #print('temp: ',temp)
                        
                        if abs(np.log(c) - mu_prior)>tolerance_level*sigma_prior :
                            temp = (abs(np.log(c) - mu_prior))/ np.sqrt(2*(np.log(-fstar_shifted+0.1)-mu_prior))
                            uncertainty = temp
                        
                            print('Not Use prior')
                
                            parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                        lengthscale_range=lengthscale_range,
                                                        variance_range=variance_range,c_range=c_range)  
                        
                    else:
                        print('sigma is big!!')
                        
                        parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                            lengthscale_range=lengthscale_range,
                                                            variance_range=variance_range,c_range=c_range) 
                            
            

                
                lengthscale = parameters[0]
                variance = parameters[1]
                c = parameters[2]
                
                # print('lengthscale: ',lengthscale)
                # print('variance: ',variance)
                print('lower bound is ',-c*Y_std+Y_min)
                            
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
        

        
    np.savetxt('exp_res/low_dimension/'+information['name']+'_SLogGP(boundary)+logTEI', LogTEI_boundary, delimiter=',')


