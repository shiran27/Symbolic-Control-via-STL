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


# initialize the example with an initial state
T = 15  #15, 20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
# scenario = ReachAvoid(x0,T)
# scenario = ReachAvoidAdv(x0,T)
# scenario = FollowTrajectory(x0,T)
scenario = FollowTrajectoryAdv(x0,T)

# Autograd based and Explicit Gradient
measureType = 4 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard
def costFunction(u,measureType=measureType):
    return scenario.costFunction(u,measureType)

# costFunctionAutoGrad = grad(costFunction)
costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()


## Set up and solve an optimization problem over u
np.random.seed(7)
u_init = np.zeros((2,T+1)).flatten()   # initial guess
u_init = np.random.rand(u_init.shape[0])


u_opt_flat = u_init
u_opt_mat = u_opt_flat.reshape((2,T+1))

# scenario.tuneKParameters(u_opt_flat, measureType, 200) # 50
scenario.tuneKParameters(u_opt_flat, measureType, 100) # 50


## SLSQP method :
start_time = time.time()
# sol = minimize(costFunction, u_init,
#         jac = costFunctionAutoGrad, # this is the function that gives the gradient (now, given by autograd library)
#         method='SLSQP')
sol = minimize(costFunction, u_init,
        jac = costFunctionExplicitGrad, # this is the function that gives the gradient (now, given by autograd library)
        method='SLSQP')
end_time = time.time()
executionTime = end_time-start_time;
print("Computation Time for SLSQP: %0.5fs" % executionTime)


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
fig, axes = plt.subplots(1)
# Plot the results
axes.set_title("SLSQP Method: Time: %1.3f \n Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n Smt: tot.R %1.3f; con.R %1.3f; obs.R %1.3f; goal.R %1.3f \n Std: tot. R %1.3f con.R %1.3f; obs.R %1.3f; goal.R %1.3f" \
    %(executionTime, costValue, controlCost, robustnessCost, robustnessCost, controlRobustness, obstacleRobustness, goalRobustness, robustnessCostStd, controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd))       
scenario.plotTrajectory(u_opt_mat,axes)
plt.show()