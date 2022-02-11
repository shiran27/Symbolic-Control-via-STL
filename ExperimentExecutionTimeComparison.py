
# THis file is to compare the executiuion time and accuracy of gradient evaluating techniques


import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad

# for data collecting
from simulations import DataCollector

# The detailed implementation of this scenario is defined here:
from scenarios import *

from statistics import *

# initialize the example with an initial state
T = 20  #15, 20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
scenario = ReachAvoid(x0,T)
# scenario = ReachAvoidAdv(x0,T)
# scenario = FollowTrajectory(x0,T)
# scenario = FollowTrajectoryAdv(x0,T)

# Autograd based and Explicit Gradient
measureType = 1 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard
def costFunction(u,measureType=measureType):
    return scenario.costFunction(u,measureType)

costFunctionAutoGrad = grad(costFunction)
costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()

def costFunctionGrad(u):
    return abs(costFunctionExplicitGrad(u)).mean()


np.random.seed(7)

u_init = np.zeros((2,T+1)).flatten()   # initial guess


numOfSimulations = 1#500
numOfTrials = 1
executionTimesExplicit = []
executionTimesAutoGrad = []
# RMSError = []
# RMSErrorNormalized = []
errors = []
sumErrorMat = []



# Initialization
# costFunctionExplicitGradGrad = grad(costFunctionGrad)
# u = u_init 
# print(costFunctionGrad(u)) 
# for i in range(20):
#     u = u + 50*costFunctionExplicitGradGrad(u)
# print(costFunctionGrad(u))
# u_init = u
# End Initialization


valErrorsEG = []
valErrorsAG = []
for rad in range(10):
    valErrorsEG.append([])  
    valErrorsAG.append([])


for i in range(numOfSimulations):

    u_opt_flat = u_init + np.random.rand(u_init.shape[0])
    
    start_time = time.time()
    for j in range(numOfTrials):
        gradValE = costFunctionExplicitGrad(u_opt_flat,measureType)
    end_time = time.time()
    et = end_time-start_time
    executionTimesExplicit.append(et/numOfTrials)

    start_time = time.time()
    for j in range(numOfTrials):
        gradValA = costFunctionAutoGrad(u_opt_flat,measureType)
    end_time = time.time()
    et = end_time-start_time
    executionTimesAutoGrad.append(et/numOfTrials)

    errorMat = np.abs(gradValE-gradValA).flatten()
    
    # print("E")
    # print(gradValE.flatten())
    # print("A")
    # print(gradValA.flatten())
    # if i==0:
    #     sumErrorMat = np.abs(errorMat)
    # else:
    #     sumErrorMat = sumErrorMat + np.abs(errorMat)
    
    # print(errorMat.mean(), errorMat.std())
    # errorMat[0] = 0
    # print(errorMat)
    # RMSE = np.sqrt((errorMat.T@errorMat).mean())
    # RMSEN = RMSE/(max(gradValA) - min(gradValA))
    # RMSError.append(RMSE)
    # RMSErrorNormalized.append(RMSEN)
    errors = np.append(errors, errorMat)
    print("Simulation: "+str(i))


    # Accuracy check 
    # costFunction
    # cost = costFunction(u_opt_flat)
    # for rad in range(10):
    #     for j in range(u_init.shape[0]):
    #         searchRadius = 0.1**rad
    #         iArray = np.zeros(u_init.shape[0]) 
    #         iArray[j] = searchRadius
    #         deltaU = iArray
    #         costIncrement = costFunction(u_opt_flat + deltaU) - cost
    #         errorEG = np.log(np.abs(costIncrement-gradValE@deltaU))
    #         errorAG = np.log(np.abs(costIncrement-gradValA@deltaU))
    #         valErrorsEG[rad].append(errorEG)
    #         valErrorsAG[rad].append(errorAG)
    

    

    

# print(executionTimesExplicit)
print(np.mean(executionTimesExplicit))

# print(executionTimesAutoGrad)
print(np.mean(executionTimesAutoGrad))
print(errors.mean(), errors.std(), np.quantile(errors, 0.90)-np.quantile(errors, 0.10))

# print(valErrorsEG)
# print(valErrorsAG)
print("Mean log absolute Error Improvement: A-E")

# print(np.mean(valErrorsEG),np.mean(valErrorsAG),logMeanAbsErrorImprovement)

# print(np.mean(valErrorsEG,axis=1))
# print(np.mean(valErrorsAG,axis=1))
# logMeanAbsErrorImprovement = np.mean(valErrorsAG,axis=1)-np.mean(valErrorsEG,axis=1)
# print(*logMeanAbsErrorImprovement)


percentageMeanExecutionTimeImprovement = 100*(np.mean(executionTimesAutoGrad)-np.mean(executionTimesExplicit))/np.mean(executionTimesAutoGrad)
# print(percentageMeanExecutionTimeImprovement)
print(np.mean(executionTimesExplicit), np.mean(executionTimesAutoGrad), percentageMeanExecutionTimeImprovement, errors.mean(), errors.std(), LA.norm(errors)/len(errors))