import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad

# for data collecting
from simulations import DataCollector

# The detailed implementation of this scenario is defined here:
from scenarios import ReachAvoid 

# initialize the example with an initial state
T = 20   #20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
scenario = ReachAvoid(x0,T)

# Autograd based and Explicit Gradient
measureType = 2 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard



## Set up and solve an optimization problem over u
np.random.seed(7)
u_init = np.zeros((2,T+1)).flatten()   # initial guess
u_init = np.random.rand(u_init.shape[0])
u_opt_flat = u_init
u_opt_mat = u_opt_flat.reshape((2,T+1))


AppEL, AppEU = scenario.getErrorBand(u_opt_mat, measureType)

AppEL, AppEU = scenario.getErrorBand(u_opt_mat, measureType, 'avoidedObstacle')
print(AppEL, AppEU)

AppEL, AppEU = scenario.getErrorBand(u_opt_mat, measureType, 'boundedControl')
print(AppEL, AppEU)

AppEL, AppEU = scenario.getErrorBand(u_opt_mat, measureType, 'reachedGoal')
print(AppEL, AppEU)