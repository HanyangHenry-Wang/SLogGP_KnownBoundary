import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

##################### GP acquisition function ########################################################
def EI(X,dim,f_best,model,f_star='no'): # X is a 2-dimensional array because we will use it in scipy.minimize

  X = X.reshape(-1,dim)

  mean,var = model.predict(X,include_likelihood=False)
  
  var[var<10**(-12)]=10**(-12)

  # z = (f_best - mean)/np.sqrt(var)        
  # out=(f_best - mean) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

  if f_star == 'no':
      z = (f_best - mean)/np.sqrt(var)        
      out=(f_best - mean) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
  else:
      z = (f_best - mean)/np.sqrt(var)        
      out1=(f_best - mean) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
      
      z = (f_star - mean)/np.sqrt(var)        
      out2=(f_star - mean) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
      
      part3 = (f_best-f_star)*norm.cdf((f_star - mean)/np.sqrt(var)  )
      
      out = out1 - out2 + part3
    
  
  return out.ravel()  #make the shape to be 1 dimensional



def EI_acquisition_opt(model,bounds,f_best,f_star='no'): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = EI(init_X,dim,f_best,model,f_star)
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -EI(X=x,dim=dim,f_best=f_best,model=model,f_star=f_star),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =  res.x  
    AF_temp = EI(X=np.array(X_temp).reshape(-1,1),dim=dim,f_best=f_best,model=model,f_star=f_star)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next

def MES(X,dim,fstar,model): 
  
  X = X.reshape(-1,dim)
  mean,var = model.predict(X,include_likelihood=False)

  var[var<10**(-12)]=10**(-12)
  gamma = -(fstar-mean)/np.sqrt(var)  

  cdf_part = norm.cdf(gamma)
  out = (gamma*norm.pdf(gamma))/(2*cdf_part)-np.log(cdf_part)

  return out.ravel() 



def MES_acquisition_opt(model,bounds,fstar): #bound should an array of size dim*2

  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = MES(init_X,dim,fstar,model)
        
    x0=init_X[np.argmax(value_holder)]


    res = minimize(lambda x: -MES(X=x,dim=dim,fstar=fstar,model=model),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

    X_temp =  res.x
    AF_temp = MES(X=np.array(X_temp).reshape(-1,1),dim=dim,fstar=fstar,model=model)
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]


  return X_next


def LCB(X,dim,model,beta): # X is a 2-dimensional array because we will use it in scipy.minimize

  X = X.reshape(-1,dim)

  mean,var = model.predict(X,include_likelihood=False)
  
  var[var<10**(-12)]=10**(-12)

  out= mean - beta*np.sqrt(var)

  return out.ravel()  #make the shape to be 1 dimensional



def LCB_acquisition_opt(model,bounds,beta): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = LCB(init_X,dim,model,beta)
      
    x0=init_X[np.argmin(value_holder)]

    res = minimize(lambda x: LCB(X=x,dim=dim,model=model,beta=beta),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =  res.x  
    AF_temp = LCB(X=np.array(X_temp).reshape(-1,1),dim=dim,model=model,beta=beta)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmin(AF_candidate)]
  lcb = np.min(AF_candidate)

  return X_next,lcb



################################# Vu's model #############################################################
def ERM(X,dim,fstar,model,mean_temp): # X is a 2-dimensional array because we will use it in scipy.minimize

  #model_temp = copy.deepcopy(model)
  X = X.reshape(-1,dim)

  mean_g,var_g = model.predict(X,include_likelihood=False)
  mean_g = mean_g + mean_temp

  var_g[var_g<10**(-12)]=10**(-12)
  sigma_g = np.sqrt(var_g)
  

  mu_f = fstar + 1/2*mean_g**2
  sigma_f = mean_g**2 * sigma_g

  gamma = (mu_f - fstar)/sigma_f      
  out=sigma_f * norm.pdf(gamma) + (mu_f - fstar) * norm.cdf(gamma)

  return out.ravel()  #make the shape to be 1 dimensional



def ERM_acquisition_opt(model,bounds,fstar,mean_temp): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = ERM(init_X,dim,fstar,model,mean_temp)
      
    x0=init_X[np.argmin(value_holder)]

    res = minimize(lambda x: ERM(X=x,dim=dim,fstar=fstar,model=model,mean_temp=mean_temp),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =  res.x      
    AF_temp = ERM(X=np.array(X_temp).reshape(-1,1),dim=dim,fstar=fstar,model=model,mean_temp=mean_temp)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmin(AF_candidate)]

  return X_next,np.min(AF_candidate)


##################### log GP acquisition function ########################################################


def SLogEI(X,dim,f_best,c,f_mean,model): # X is a 2-dimensional array because we will use it in scipy.minimize


  X = X.reshape(-1,dim)

  mean,var = model.predict(X,include_likelihood=False)  
  var[var<10**(-12)]=10**(-12)
  sigma = np.sqrt(var)
  mu = mean+f_mean

  C = c+f_best
  
  out = C*norm.cdf((np.log(C)-mu)/sigma)-np.exp(mu+sigma**2/2)*norm.cdf((np.log(C)-mu-sigma**2)/sigma)

  return out.ravel()  #make the shape to be 1 dimensional



def SLogEI_acquisition_opt(model,bounds,f_best,c,f_mean): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder =  SLogEI(init_X,dim,f_best,c,f_mean,model)
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -SLogEI(X=x,dim=dim,f_best=f_best,c=c,f_mean=f_mean,model=model),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =   res.x  
    AF_temp = SLogEI(X=np.array(X_temp).reshape(-1,1),dim=dim,f_best=f_best,c=c,f_mean=f_mean,model=model)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next


def SLogTEI(X,dim,f_best,c,f_mean,fstar,model): 
  
  X = X.reshape(-1,dim)

  mean,var = model.predict(X,include_likelihood=False)  
  var[var<10**(-12)]=10**(-12)
  sigma = np.sqrt(var)
  mu = mean+f_mean
  
  C = c+f_best
  part1 = C*norm.cdf((np.log(C)-mu)/sigma)-np.exp(mu+sigma**2/2)*norm.cdf((np.log(C)-mu-sigma**2)/sigma)
  
  C = c+fstar
  part2 = C*norm.cdf((np.log(C)-mu)/sigma)-np.exp(mu+sigma**2/2)*norm.cdf((np.log(C)-mu-sigma**2)/sigma)
  
  
  part3 = (f_best-fstar)*norm.cdf(  (np.log(fstar+c)-mu) /sigma ) 
  
  out = part1-part2+part3
  
  return out.ravel()  #make the shape to be 1 dimensional


def SLogTEI_acquisition_opt(model,bounds,f_best,c,f_mean,fstar): #bound should an array of size dim*2
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder =  SLogTEI(init_X,dim,f_best,c,f_mean,fstar,model)
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -SLogTEI(X=x,dim=dim,f_best=f_best,c=c,f_mean=f_mean,fstar=fstar,model=model),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) 

    X_temp =   res.x  
    AF_temp = SLogTEI(X=np.array(X_temp).reshape(-1,1),dim=dim,f_best=f_best,c=c,f_mean=f_mean,fstar=fstar,model=model)
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next




