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
T = 20  #20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
# scenario = ReachAvoid(x0,T)
# scenario = ReachAvoidAdv(x0,T)
# scenario = FollowTrajectory(x0,T)
scenario = FollowTrajectoryAdv(x0,T)

fig, axes = plt.subplots(1)
scenario.draw(axes)


measureType = 4 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard
def costFunction(u,measureType=measureType):
    return scenario.costFunction(u,measureType)

costFunctionAutoGrad = grad(costFunction)

costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()

## Set up and solve an optimization problem over u
np.random.seed(7)
u_init = np.zeros((2,T+1)).flatten()   # initial guess
u_init = np.random.rand(u_init.shape[0])
u_opt_flat = u_init
u_opt_mat = u_opt_flat.reshape((2,T+1))

gradVal = costFunctionAutoGrad(u_opt_flat,measureType)
print(gradVal)

# Plot the results
figs, axes = plt.subplots(1,2)
scenario.plotTrajectory(u_opt_mat,axes[0])




I = np.eye(2)
O = np.zeros((2,2))
B = np.block([[I if t>tau and tau>0 else O for tau in range(T+1)] for t in range(T+1)])
print(B)
# print(B.T)


plt.show()