### This file has been written assuming the agent to be a quadcopter


## Basic libraries
import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad


## For data collecting
from simulations import DataCollector


## The detailed implementation of this scenario is defined here:
# from scenarios import *         # for the  single integrator
from quad_scenarios import *    # for the quadcopter
# from doub_scenarios import *    # for the quadcopter


## Initialize the example with an initial state
T = 20  #16, 20 number of timesteps

x0 = np.array([0,0,0,2,15,0])[:,np.newaxis] # For ReachAvoid class        # Quadcopter
# x0 = np.array([0,10,0,2,10,0])[:,np.newaxis] # For ReachAvoid_Zihao class   # Quadcopter
# x0 = np.array([0,0,0,0])[:,np.newaxis] # For ReachAvoid class        # Double Integrator
#x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4             # Single Integrator


## Loading the scenario
scenario = ReachAvoid(x0,T)
# scenario = ReachAvoid_Zihao(x0,T)
# scenario = ReachAvoidAdv(x0,T)
# scenario = FollowTrajectory(x0,T)
# scenario = FollowTrajectoryAdv(x0,T)


## Robustness measure type
measureType = 3 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard


## Cost function
def costFunction(u,measureType=measureType):
    return scenario.costFunction(u,measureType)


## Autograd based and Explicit cost function gradient
costFunctionAutoGrad = grad(costFunction)
# costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()


## Set up the initial control input signal
np.random.seed(7)
u_init = np.zeros((2,T+1)).flatten()   # initial guess
u_init = np.random.uniform(-10, 10,( u_init.shape[0] ))
#u_init = np.random.rand(u_init.shape[0])

u_opt_flat = u_init
u_opt_mat = u_opt_flat.reshape((2,T+1))


## Smooth robustness measure parameter tuning
# scenario.tuneKParameters(u_opt_flat, measureType, 200) # 50
# scenario.tuneKParameters(u_opt_flat, measureType, 100) # 50


## Plotting the initial trajectory
# fig, axes = plt.subplots(1)
# scenario.plotTrajectory(u_opt_mat,axes)


## Solve the optimization problem using the SLSQP method :
start_time = time.time()
sol = minimize(costFunction, u_init,
        jac = costFunctionAutoGrad, # this is the function that gives the gradient (now, given by autograd library)
        method='SLSQP')
# sol = minimize(costFunction, u_init,
#         jac = costFunctionExplicitGrad, # this is the function that gives the gradient 
#         method='SLSQP')
end_time = time.time()
executionTime = end_time-start_time;
print("Computation Time for SLSQP: %0.5fs" % executionTime)


## Cost and robustness value analysis
costValue = scenario.costFunction(sol.x, measureType)
u_opt_mat = sol.x.reshape((2,T+1))
u_opt_flat = sol.x

# scenario.plotControlProfile(u_opt_mat, axesu[2], "SLSQP")
robustnessCost = scenario.getRobustness(u_opt_mat, measureType)
controlCost = robustnessCost+costValue

controlRobustness = scenario.getRobustness(u_opt_mat, measureType, "boundedControl")
obstacleRobustness = scenario.getRobustness(u_opt_mat, measureType, "avoidedObstacle")
goalRobustness = scenario.getRobustness(u_opt_mat, measureType, "reachedGoal")

# Standard robustness measures
robustnessCostStd = scenario.getRobustness(u_opt_mat, 0)
controlRobustnessStd = scenario.getRobustness(u_opt_mat, 0, "boundedControl")
obstacleRobustnessStd = scenario.getRobustness(u_opt_mat, 0, "avoidedObstacle")
goalRobustnessStd = scenario.getRobustness(u_opt_mat, 0, "reachedGoal")

errorBand = scenario.getErrorBand(u_opt_mat, measureType)
print("Error Band: %1.4f, %1.4f in [%1.4f, %1.4f]"%(robustnessCostStd,robustnessCost,robustnessCost+errorBand[0],robustnessCost+errorBand[1]))
print("Cost: %1.3f; con.C: %1.3f; rob.C: %1.3f; con.R %1.3f; obs.R %1.3f; Goal.R %1.3f" %(costValue,controlCost,robustnessCost,controlRobustness,obstacleRobustness,goalRobustness))



## Plotting the final result
fig, axes = plt.subplots(1)
# Plot the results
axes.set_title("SLSQP Method: Time: %1.3f \n Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n Smt: tot.R %1.3f; con.R %1.3f; obs.R %1.3f; goal.R %1.3f \n Std: tot. R %1.3f con.R %1.3f; obs.R %1.3f; goal.R %1.3f" \
    %(executionTime, costValue, controlCost, robustnessCost, robustnessCost, controlRobustness, obstacleRobustness, goalRobustness, robustnessCostStd, controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd))       
scenario.plotTrajectory(u_opt_mat,axes)
plt.show()