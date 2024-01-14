import botorch
from known_boundary.acquisition_function import EI,SLogEI,EI_acquisition_opt,MES_acquisition_opt,LCB_acquisition_opt,ERM_acquisition_opt,SLogTEI_acquisition_opt,SLogEI_acquisition_opt
from known_boundary.utlis import  get_initial_points,transform,opt_model_MLE,opt_model_MAP
import numpy as np
import GPy
from sklearn.metrics.pairwise import euclidean_distances
import torch
from botorch.test_functions import Ackley,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,DixonPrice,Levy,StyblinskiTang,Griewank
import obj_functions.push_problems
from botorch.utils.transforms import unnormalize,normalize
from known_boundary.SLogGP import SLogGP
import scipy 
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []
fstar=0.397887 
distances = np.array([0.001,0.1,1,2,5,10,50,100,200])

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
temp['name']='Hartmann3D' 
temp['function'] = Hartmann(dim=3,negate=False)
temp['fstar'] =  -3.86278
function_information.append(temp)


for information in function_information:

    total_record = []
    total_record2 = []

    for dist in distances:
        
        
        print('!!!!distance is: ',dist)
        
    

        fun = information['function']
        dim = fun.dim
        bounds = fun.bounds
        standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
        
        n_init = 4*dim

        
        # fstar = dist #information['fstar']
        
        # print('fstar is: ',fstar)
        

        step_size = 2
        iter_num = 1
        N = 50

        lengthscale_range = [0.001,2]
        variance_range = [0.001**2,20]
        noise = 1e-6
        
        print(information['name'])
        ratio_holder = []
        dis_holder = []
        ######################## SlogGP+logEI#######################################

        for exp in range(N):

            seed = exp
        
            print(exp)

            X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        
        
            Y_BO = torch.tensor(
                [fun(x) for x in X_BO], dtype=dtype, device=device
            ).reshape(-1,1)



            best_record = [Y_BO.min().item()]
            np.random.seed(1234)
        
            boundarys = []
            variances = []


        
            i = 0
            print('inner loop: ',i)

            
            train_Y = Y_BO.numpy()
            
            print('so far best: ',np.min(train_Y))
            fstar = np.min(train_Y) - dist
            
            if i%step_size == 0  :
                Y_min = np.min(train_Y)
                Y_std = np.std(train_Y-Y_min)
                
            fstar_shifted = fstar -Y_min # shifted lower bound
            train_Y = train_Y - Y_min  # shift Y
            
            #scalise Y_shift and fstar_shift
            train_Y = train_Y/Y_std
            fstar_shifted = fstar_shifted/Y_std
        

            
            train_X = normalize(X_BO, bounds)
            train_X = train_X.numpy()
            
            lower = -fstar_shifted
            upper = 10**(-6)+lower
            
        
                
            c_range = [lower,upper]

            if i%step_size == 0:
                
                parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range,c_range=c_range)                

                lengthscale = parameters[0]
                variance = parameters[1]
                c = parameters[2]
            
            print('lengthscale is ',lengthscale)
            print('variance is ',variance)
            print('lower bound is ',-c*Y_std+Y_min)
            
                
            boundarys.append(-c*Y_std+Y_min)
            variances.append(variance)
            
            
            warp_Y = np.log(train_Y+c)
            mean_warp_Y = np.mean(warp_Y) # use to predict mean
            warp_Y_standard = warp_Y-mean_warp_Y
            
            
            kernel = GPy.kern.RBF(input_dim=dim,lengthscale= lengthscale,variance=variance)  
            m = GPy.models.GPRegression(train_X, warp_Y_standard,kernel)
            m.Gaussian_noise.variance.fix(noise)
            
            np.random.seed(i)
            standard_next_X = SLogEI_acquisition_opt(model=m,bounds=standard_bounds,f_best=np.min(train_Y),c=c,
                                                    f_mean=mean_warp_Y)
            
            best_idx = np.argmin(train_Y)
            best_X = train_X[best_idx]
            
            dis = euclidean_distances(standard_next_X.reshape(-1,dim),best_X.reshape(-1,dim))
            print('dis is: ',dis)
            dis_holder.append(dis)
            
            
            
            # parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range) 
            # lengthscale = parameters[0]
            # variance = parameters[1]
        
            # kernel = GPy.kern.RBF(input_dim=dim,lengthscale=lengthscale,variance=variance)
            # m = GPy.models.GPRegression(train_X.reshape(-1,dim), train_Y.reshape(-1,1),kernel)
            # m.Gaussian_noise.fix(noise)
            
            # acq_val = EI(X=standard_next_X,dim=dim,f_best=np.min(train_Y),model=m,f_star='no')
            # mean,var = m.predict_noiseless(standard_next_X.reshape(-1,dim))
            # f_best = np.min(train_Y)
            # z = (f_best - mean)/np.sqrt(var)        
            # out=(f_best - mean) * norm.cdf(z)
            # ratio = out/acq_val[0]
            # ratio_holder.append(ratio)
            # print('acquisition is ',acq_val[0])
            # print('exploit is ',out)
            # print('ratio is: ',ratio)
        #     acq_val = SLogEI(X=standard_next_X,dim=dim,f_best=np.min(train_Y),c=c,f_mean=mean_warp_Y,model=m)
        #     mean,var = m.predict_noiseless(standard_next_X.reshape(-1,dim))
        #     mu = mean+mean_warp_Y
        #     predictMean = np.exp(mu+0.5*var)-c
        #     print('mean is ',predictMean)
        #     print('so far best is: ',np.min(train_Y) )
        #     print('acqusition is: ',acq_val)
        #     ratio = (np.min(train_Y) - predictMean)/acq_val
        #     ratio_holder.append(ratio)
            
        # total_record2.append(ratio_holder)
        total_record.append(dis_holder)
        
    total_record = np.array(total_record).reshape(-1,N)
    np.savetxt('distance/'+information['name']+'_SLogGP+logEI_distance', total_record, delimiter=',')
    