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
from scenarios import ReachAvoidAdv 

# initialize the example with an initial state
T = 20   #20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
# scenario = ReachAvoid(x0,T)
scenario = ReachAvoidAdv(x0,T)

# Autograd based and Explicit Gradient
measureType = 2 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard
def costFunction(u):
    return scenario.costFunction(u,measureType)

costFunctionAutoGrad = grad(costFunction)
costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()


## Set up and solve an optimization problem over u
np.random.seed(7)
u_init = np.zeros((2,T+1)).flatten()   # initial guess
u_init = np.random.rand(u_init.shape[0])


## A simple gradient scheme:
numOfSteps = 200  
stepSize = 0.25 #0.25, 0.5
precision = 0.01
u_opt_flat = u_init
u_opt_mat = u_opt_flat.reshape((2,T+1))

# plotting control
figu, axesu = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
scenario.plotControlProfile(u_opt_mat, axesu[0], "Initial")
plt.show(block = False)
# axu.plot(u_opt_mat[0,:].tolist(), u_opt_mat[1,:].tolist(), label="Initial", linestyle="-", marker=".")


### Lets use a data collection mechanism with preallocated everything
names = ["Total Cost","Control Cost","Robustness Cost","Control Robustness","Obstacle Robustness","Goal Robustness",\
         "Std. Total Robustness", "Std. Control Robustness","Std. Obstacle Robustness","Std. Goal Robustness"]
shortNames = ["Tot. C.", "Con. C.", "Rob. C.","Con. R.","Obs. R.","Goal R.","Std. Rob. C.","Std. Con. R.","Std. Obs. R.","Std. Goal. R."]
dataset =  DataCollector(numOfSteps,names,shortNames)

shortNames2 = ["EGradN","AGradN","RMSE","RMSEN","Approx. Rob.","Err. L.","Err. H.","Std. Rob.", "Std. Err. L.","Std. Err. H."]
dataset2 = DataCollector(numOfSteps,shortNames2)

shortNames3 = ["Con. R.","Con. L.","Con. H.", "Obs. R.","Obs. L.","Obs. H.", "Goal. R.","Goal. L.","Goal. H.",\
               "Std. Con. R.","Std. Con. L.","Std. Con. H.", "Std. Obs. R.","Std. Obs. L.","Std. Obs. H.", "Std. Goal. R.","Std. Goal. L.","Std. Goal. H."]
dataset3 = DataCollector(numOfSteps,shortNames3)
### End Data

stepNumber = 0
boost_mode = 0
fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=False)

start_time = time.time()
for i in range(numOfSteps):

    if boost_mode==0 or boost_mode==2:
        gradVal = costFunctionExplicitGrad(u_opt_flat)
        # gradVal = costFunctionAutoGrad(u_opt_flat)
    elif boost_mode==1:
        gradVal = costFunctionExplicitGrad(u_opt_flat,measureType+1)

    if boost_mode==0 and LA.norm(gradVal)<precision:
        print("Local optimum reached at iteration %i" %i)
        boost_mode = 1; # use 2; to avoid boosting 
        stepNumber = 0;
        # break
    elif boost_mode==1 and LA.norm(gradVal)<precision:
        print("Boost Mode Ended at iteration %i" %i)
        boost_mode = 2;
        stepNumber = 0;

    elif boost_mode==2 and LA.norm(gradVal)<precision:
        print("Local optimum reached at iteration %i" %i)
        boost_mode = 0;
        stepNumber = 0;
        break

    u_opt_flat = u_opt_flat - (stepSize/np.power(stepNumber+1,0.25))*gradVal
    stepNumber = stepNumber + 1



    ### Start Data Collecting
    ## First dataset: ["Tot. C.", "Con. C.", "Rob. C.","Con. R.","Obs. R.","Goal R.","Std. Rob. C.","Std. Con. R.","Std. Obs. R.","Std. Goal. R."]
    costValue = scenario.costFunction(u_opt_flat, measureType)
    u_opt_mat = u_opt_flat.reshape((2,T+1))
    robustnessCost = scenario.getRobustness(u_opt_mat, measureType)
    controlCost = robustnessCost+costValue

    controlRobustness = scenario.getRobustness(u_opt_mat, measureType, "boundedControl")
    obstacleRobustness = scenario.getRobustness(u_opt_mat, measureType, "avoidedObstacle")
    goalRobustness = scenario.getRobustness(u_opt_mat, measureType, "reachedGoal")
    
    robustnessCostStd = scenario.getRobustness(u_opt_mat, 0)
    controlRobustnessStd = scenario.getRobustness(u_opt_mat, 0, "boundedControl")
    obstacleRobustnessStd = scenario.getRobustness(u_opt_mat, 0, "avoidedObstacle")
    goalRobustnessStd = scenario.getRobustness(u_opt_mat, 0, "reachedGoal")
    
    dataArray = [costValue, controlCost, robustnessCost, controlRobustness, obstacleRobustness, goalRobustness,\
                 robustnessCostStd, controlRobustnessStd, obstacleRobustnessStd, goalRobustnessStd]
    dataset.updateDataset(i,dataArray)
    dataset.printDataset(i, 'Boost: '+str(boost_mode))
    
    
    ## Second data set: ["EGradN","AGradN","RMSE","RMSEN","Approx. Rob.","Err. L.","Err. H.","Std. Rob.", "Std. Err. L.","Std. Err. H."]
    gradValE = costFunctionExplicitGrad(u_opt_flat)
    gradValA = costFunctionAutoGrad(u_opt_flat)
    errorMat = (gradValE-gradValA).flatten()
    EGradN = LA.norm(gradValE) 
    AGradN = LA.norm(gradValA) 
    RMSE = np.sqrt((errorMat.T@errorMat).mean())
    RMSEN = RMSE*100/(max(gradValA) - min(gradValA))
    
    AppRob = robustnessCost
    AppEL, AppEU = scenario.getErrorBand(u_opt_mat, measureType)
    StdRob = robustnessCostStd
    StdEL, StdEU = scenario.getErrorBand(u_opt_mat, 0)
    
    dataArray2 = [EGradN,AGradN,RMSE,RMSEN,AppRob,AppEL+AppRob,AppEU+AppRob,StdRob,StdEL+StdRob,StdEU+StdRob]
    dataset2.updateDataset(i,dataArray2)
    # dataset2.printDataset(i,'')


    ## Third data set: ["Con. R.","Con. L.","Con. H.", "Obs. R.","Obs. L.","Obs. H.", "Goal. R.","Goal. L.","Goal. H.",
                        #"Std. Con. R.","Std. Con. L.","Std. Con. H.", "Std. Obs. R.","Std. Obs. L.","Std. Obs. H.", "Std. Goal. R.","Std. Goal. L.","Std. Goal. H."]
    ConEL, ConEU = scenario.getErrorBand(u_opt_mat, measureType, 'boundedControl')
    ObsEL, ObsEU = scenario.getErrorBand(u_opt_mat, measureType, 'avoidedObstacle')
    GoalEL, GoalEU = scenario.getErrorBand(u_opt_mat, measureType, 'reachedGoal')
    ConRob, ObsRob, GoalRob = controlRobustness, obstacleRobustness, goalRobustness
    StdConEL, StdConEU = scenario.getErrorBand(u_opt_mat, 0, 'boundedControl')
    StdObsEL, StdObsEU = scenario.getErrorBand(u_opt_mat, 0, 'avoidedObstacle')
    StdGoalEL, StdGoalEU = scenario.getErrorBand(u_opt_mat, 0, 'reachedGoal')
    StdConRob, StdObsRob, StdGoalRob = controlRobustnessStd, obstacleRobustnessStd, goalRobustnessStd
    
    dataArray3 = [ConRob, ConRob+ConEL, ConRob+ConEU, ObsRob, ObsRob+ObsEL, ObsRob+ObsEU, GoalRob, GoalRob+GoalEL, GoalRob+GoalEU,\
                StdConRob, StdConRob+StdConEL, StdConRob+StdConEU, StdObsRob, StdObsRob+StdObsEL, StdObsRob+StdObsEU, StdGoalRob, StdGoalRob+StdGoalEL, StdGoalRob+StdGoalEU]
    dataset3.updateDataset(i,dataArray3)
    # dataset3.printDataset(i,'')
    ### End Data Collecting

    if i % 20==0:
        # Plot the results
        if int(i/20) <= 7 :
            plotNumber = int(i/20)
            if plotNumber<=3:
                ax = axes[0,plotNumber]
            else:
                ax = axes[1,plotNumber-4]
            ax.set_title("i: %i; Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n con.R %1.3f; obs.R %1.3f; Goal.R %1.3f" %(i,costValue,controlCost,robustnessCost,controlRobustness,obstacleRobustness,goalRobustness))
            
            scenario.plotTrajectory(u_opt_mat, ax)
            plt.xlim([-1,10])
            # fig.canvas.draw()
            plt.show(block = False)
            

end_time = time.time()
executionTime = (end_time-start_time);
numOfStepsRan = i
dataset.truncate(numOfStepsRan)
dataset2.truncate(numOfStepsRan)
dataset3.truncate(numOfStepsRan)

scenario.plotControlProfile(u_opt_mat, axesu[1], "GD")
u_opt_mat = u_opt_flat.reshape((2,T+1))
print("Computation Time for GD: %0.5fs" %executionTime)

### End Data
## cost convergence plot
figs, axes = plt.subplots(2,3,sharex=True,sharey=False)
axes[0,0].set_title("Total Cost")
axes[0,0].plot(range(numOfStepsRan),dataset.dataset[0], linestyle="-", color="r")
axes[0,1].set_title("Control Cost")
axes[0,1].plot(range(numOfStepsRan),dataset.dataset[1], linestyle="-", color="r")
axes[0,2].set_title("Robustness Cost")
axes[0,2].plot(range(numOfStepsRan),dataset.dataset[2], linestyle="-", color="r", label="Smooth")
axes[0,2].plot(range(numOfStepsRan),dataset.dataset[6], linestyle="-", color="k", label="Non-Smooth")
axes[0,2].legend()
axes[1,0].set_title("Control Robustness")
axes[1,0].plot(range(numOfStepsRan),dataset.dataset[3], linestyle="-", color="r")
axes[1,0].plot(range(numOfStepsRan),dataset.dataset[7], linestyle="-", color="k")
axes[1,1].set_title("Obstacle Robustness")
axes[1,1].plot(range(numOfStepsRan),dataset.dataset[4], linestyle="-", color="r")
axes[1,1].plot(range(numOfStepsRan),dataset.dataset[8], linestyle="-", color="k")
axes[1,1].set_xlabel("Gradient Ascent Iteration")
axes[1,2].set_title("Goal Robustness")
axes[1,2].plot(range(numOfStepsRan),dataset.dataset[5], linestyle="-", color="r")
axes[1,2].plot(range(numOfStepsRan),dataset.dataset[9], linestyle="-", color="k")


## For the second dataset
figs, axes = plt.subplots(1,2)
axes[0].set_title("Explict Gradient vs. 'AutoGrad' Gradient")
axes[0].plot(range(numOfStepsRan),dataset2.dataset[0], linestyle="-", color="r", label="Explict")
axes[0].plot(range(numOfStepsRan),dataset2.dataset[1], linestyle="-", color="k", label="'AutoGrad'")
axes[0].set_xlabel("Gradient Ascent Iteration")
axes[0].legend()
# axes[1].set_title("Normalized RMS Error")
# axes[1].plot(range(numOfStepsRan),dataset2.dataset[3], linestyle="-", color="g")
axes[1].set_title("Robustness Measure")
axes[1].plot(range(numOfStepsRan),dataset2.dataset[4], linestyle="-", color="r", label="Smooth")
axes[1].fill_between(range(numOfStepsRan), dataset2.dataset[5], dataset2.dataset[6], alpha=0.2, color="r")
axes[1].plot(range(numOfStepsRan),dataset2.dataset[7], linestyle="-", color="k", label="Non-Smooth")
axes[1].fill_between(range(numOfStepsRan), dataset2.dataset[8], dataset2.dataset[9], alpha=0.2, color="k")
axes[1].set_xlabel("Gradient Ascent Iteration")
axes[1].legend()


## For the third dataset: ["Con. R.","Con. L.","Con. H.", "Obs. R.","Obs. L.","Obs. H.", "Goal. R.","Goal. L.","Goal. H.",
                        #"Std. Con. R.","Std. Con. L.","Std. Con. H.", "Std. Obs. R.","Std. Obs. L.","Std. Obs. H.", "Std. Goal. R.","Std. Goal. L.","Std. Goal. H."]
figs, axes = plt.subplots(1,3,sharex=True,sharey=False)
axes[0].set_title("Control Robustness Measure")
axes[0].plot(range(numOfStepsRan),dataset3.dataset[0], linestyle="-", color="r")
axes[0].fill_between(range(numOfStepsRan), dataset3.dataset[1], dataset3.dataset[2], alpha=0.2, color="r")
axes[0].plot(range(numOfStepsRan),dataset3.dataset[9], linestyle="-", color="k")
axes[0].fill_between(range(numOfStepsRan), dataset3.dataset[10], dataset3.dataset[11], alpha=0.2, color="k")
axes[1].set_title("Obstacle Robustness Measure")
axes[1].plot(range(numOfStepsRan),dataset3.dataset[3], linestyle="-", color="r", label="Smooth")
axes[1].fill_between(range(numOfStepsRan), dataset3.dataset[4], dataset3.dataset[5], alpha=0.2, color="r")
axes[1].plot(range(numOfStepsRan),dataset3.dataset[12], linestyle="-", color="k", label="Non-Smooth")
axes[1].fill_between(range(numOfStepsRan), dataset3.dataset[13], dataset3.dataset[14], alpha=0.2, color="k")
axes[1].set_xlabel("Gradient Ascent Iteration")
axes[1].legend()
axes[2].set_title("Goal Robustness Measure")
axes[2].plot(range(numOfStepsRan),dataset3.dataset[6], linestyle="-", color="r")
axes[2].fill_between(range(numOfStepsRan), dataset3.dataset[7], dataset3.dataset[8], alpha=0.2, color="r")
axes[2].plot(range(numOfStepsRan),dataset3.dataset[15], linestyle="-", color="k")
axes[2].fill_between(range(numOfStepsRan), dataset3.dataset[16], dataset3.dataset[17], alpha=0.2, color="k")
### End Data




# Plot the results
figs, axes = plt.subplots(1,2)
# axes[0].set_title("GD Method: Time: %1.3f \n i: %i; Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n con.R %1.3f; obs.R %1.3f; Goal.R %1.3f" %(executionTime,i,costValue,controlCost,robustnessCost,controlRobustness,obstacleRobustness,goalRobustness))       
axes[0].set_title("GD Method: Time: %1.3f \n Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n Smt: tot.R %1.3f; con.R %1.3f; obs.R %1.3f; goal.R %1.3f \n Std: tot. R %1.3f con.R %1.3f; obs.R %1.3f; goal.R %1.3f" \
  %(executionTime, costValue, controlCost, robustnessCost, robustnessCost, controlRobustness, obstacleRobustness, goalRobustness, robustnessCostStd, controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd))       
scenario.plotTrajectory(u_opt_mat,axes[0])




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
scenario.plotControlProfile(u_opt_mat, axesu[2], "SLSQP")
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

print("Cost: %1.3f; con.C: %1.3f; rob.C: %1.3f; con.R %1.3f; obs.R %1.3f; Goal.R %1.3f" %(costValue,controlCost,robustnessCost,controlRobustness,obstacleRobustness,goalRobustness))
    
# Plot the results
axes[1].set_title("SLSQP Method: Time: %1.3f \n Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n Smt: tot.R %1.3f; con.R %1.3f; obs.R %1.3f; goal.R %1.3f \n Std: tot. R %1.3f con.R %1.3f; obs.R %1.3f; goal.R %1.3f" \
    %(executionTime, costValue, controlCost, robustnessCost, robustnessCost, controlRobustness, obstacleRobustness, goalRobustness, robustnessCostStd, controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd))       
scenario.plotTrajectory(u_opt_mat,axes[1])
plt.show()