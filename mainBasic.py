import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad


# The detailed implementation of this scenario is defined here:
from scenarios import ReachAvoid 
from scenarios import ReachAvoidAdv 


# initialize the example with an initial state
T = 20   #20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
scenario = ReachAvoid(x0,T)
scenario = ReachAvoidAdv(x0,T)



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

# print("AutoGrad")
# costAutoGrad = gradVal.reshape(2,T+1)
# print(costAutoGrad)

# print("ExplicitGrad")
# costExplicitGrad = scenario.getRobustnessGrad(u_test, measureType)
# print(costExplicitGrad)

# print("RMSError")
# errorMat = (costExplicitGrad-costAutoGrad).flatten()
# rmsError = np.sqrt((errorMat.T@errorMat).mean())
# print(rmsError)

# print(scenario.fullSpec.STLFormulaObjects[measureType].parameters)
# print(scenario.fullSpec.STLFormulaObjects[measureType].parameters)


## Draw Graphs
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
# fig1, ax1 = plt.subplots(1)
scenario.plotTrajectory(u_test,ax1)




# debugging control constraint related gradient in SA case
measureType = 0
signal = scenario.getSignal(u_test)
# print(signal)
para = scenario.boundedControl.STLFormulaObjects[measureType].parameters
# print(para)
# print(scenario.boundedControl.STLFormulaObjects[measureType].robustness(signal.T, 0, para)) # this value (-0.81229) is wrong

# print(scenario.controlBounds.isPointInside(signal[2,:],signal[3,:])) # robustness according to this 0.022 at t = 4, which is correct
# Mu, i, j =  [1.024899227550348, 3, 19], errorneous value: 1.025
# figu, axesu = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
# scenario.plotControlProfile(u_test, axesu[0], "Initial")

print(signal[0:2,:])
print(scenario.regions[0].isPointInside(signal[0,:],signal[1,:])) # robustness according to this -0.2002 at t = 10, which is correct
# Mu, i, j =  [6.548971863810452, 3, 20], erroneous robustness value = -6.549

# print(scenario.regions[1].isPointInside(signal[0,:],signal[1,:]))


robustnessCost = scenario.getRobustness(u_test, measureType)
controlRobustness = scenario.getRobustness(u_test, measureType, "boundedControl")
obstacleRobustness = scenario.getRobustness(u_test, measureType, "avoidedObstacle")
goalRobustness = scenario.getRobustness(u_test, measureType, "reachedGoal")

print("rob.C: %1.5f; con.R %1.5f; obs.R %1.5f; Goal.R %1.5f" %(robustnessCost,controlRobustness,obstacleRobustness,goalRobustness))





# print([scenario.boundedControl.STLFormulaObjects[measureType].robustness(signal.T, tau, para) for tau in range(0, T+1)])
# print([scenario.avoidedObstacle.STLFormulaObjects[measureType].robustness(signal.T, tau, para) for tau in range(0, T+1)])


# print(scenario.avoidedObstacle.STLFormulaObjects[measureType].robustness(signal.T, 0, para)) # this value (-0.81229) is wrong
# print(scenario.reachedGoal.STLFormulaObjects[measureType].robustness(signal.T, 0, para)) # this value (-0.81229) is wrong



plt.show() 