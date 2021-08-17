import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad


# The detailed implementation of this scenario is defined here:
from scenarios import ReachAvoid 


# initialize the example with an initial state
T = 20   								# 20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] 	# [0.5,5] for ReachAvoid4
scenario = ReachAvoid(x0,T)



# # Set up and solve an optimization problem over u
np.random.seed(7)
u_test = np.zeros((2,T+1)).flatten()   # initial guess
u_test = np.random.rand(u_test.shape[0])

measureType = 0


u_test = u_test.reshape((2,T+1))
signal = scenario.getSignal(u_test)


print(signal[0:2,:])
# print(signal)
print(scenario.regions[0].isPointInside(signal[0,:],signal[1,:])) # robustness according to this -0.2002 at t = 10, which is correct
# Mu, i, j =  [6.548971863810452, 3, 20], erroneous robustness value = -6.549


# from robustnessMeasures.STLFormulaNonSmooth import STLFormulaNS
# print("TT1", [scenario.regions[0].predicates[i](signal[0,T],signal[1,T]) for i in range(4)])
# print("TT2", STLFormulaNS.minFun([scenario.regions[0].predicates[i](signal[0,T],signal[1,T]) for i in range(4)]))
# print("TT3", [-1*STLFormulaNS.minFun([scenario.regions[0].predicates[i](signal[0,k],signal[1,k]) for i in range(4)]) for k in range(T+1)])
# print("TT4", STLFormulaNS.minFun([-1*STLFormulaNS.minFun([scenario.regions[0].predicates[i](signal[0,k],signal[1,k]) for i in range(4)]) for k in range(T+1)]))


obstacleRobustness = scenario.getRobustness(u_test, measureType, "avoidedObstacle")
print("obs.R %1.5f" %(obstacleRobustness))


# para = scenario.avoidedObstacle.STLFormulaObjects[measureType].parameters
# print(scenario.avoidedObstacle.STLFormulaObjects[measureType].robustness(signal.T, 0, para)) # this value (-1.799) is wrong
# print([scenario.avoidedObstacle.STLFormulaObjects[measureType].robustness(signal.T, k, para) for k in range(1)])

# read the anonymous function parameters stored inside


# print([scenario.avoidedObstacle.STLFormulaObjects[measureType].robustness(signal.T, k, para) for k in range(0, T+1)])
# print(STLFormulaNS.minFun([scenario.avoidedObstacle.STLFormulaObjects[measureType].robustness(signal.T, k, para) for k in range(0, T+1)]))






fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
# fig1, ax1 = plt.subplots(1)
scenario.plotTrajectory(u_test,ax1)
plt.show() 
