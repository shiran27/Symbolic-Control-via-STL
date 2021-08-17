import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad


# The detailed implementation of this scenario is defined here:
from scenarios import ReachAvoid 


# initialize the example with an initial state
T = 20   #20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
scenario = ReachAvoid(x0,T)



# # Set up and solve an optimization problem over u
np.random.seed(7)
u_test = np.zeros((2,T+1)).flatten()   # initial guess
u_test = np.random.rand(u_test.shape[0])

measureType = 1

costFunctionGrad = grad(scenario.costFunction)
gradVal = costFunctionGrad(u_test, measureType)

print("Cost, Robustness, ErrorBand:")
costValue = scenario.costFunction(u_test, measureType)
u_test = u_test.reshape((2,T+1))
robustness = scenario.getRobustness(u_test, measureType)
errorBand = scenario.getErrorBand(u_test, measureType)
print(costValue, robustness, errorBand)

print("AutoGrad")
costAutoGrad = gradVal.reshape(2,T+1)
print(costAutoGrad)

print("ExplicitGrad")
costExplicitGrad = scenario.getRobustnessGrad(u_test, measureType)
print(costExplicitGrad)

print("RMSError")
errorMat = (costExplicitGrad-costAutoGrad).flatten()
rmsError = np.sqrt((errorMat.T@errorMat).mean())
print(rmsError)

# print(scenario.fullSpec.STLFormulaObjects[measureType].parameters)
# print(scenario.fullSpec.STLFormulaObjects[measureType].parameters)


## Draw Graphs
fig1, ax1 = plt.subplots(1)
scenario.plotTrajectory(u_test,ax1)
plt.show() 