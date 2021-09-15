import time
from numpy import linalg as LA 
from numpy import save
from numpy import load
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad

# for data collecting
from simulations import DataCollector

# The detailed implementation of this scenario is defined here:
from scenarios import *


# initialize the example with an initial state
T = 20  #15, 20 number of timesteps


numOfRealizations = 50
dataMatrix = [] #each row: [realization,totalcost,robustness,controlcost,executiontime,r_1,r_2,r_3]

meanStdRobustness = 0
meanStdRobustnessTemp = 0

for realization in range(numOfRealizations):
    print("Realization %i started. Mean robustness: %1.4f" % (realization,meanStdRobustnessTemp))
    np.random.seed(realization+1)

    x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
    # x0 = np.asarray(np.random.rand(2))[:,np.newaxis]

    scenario = ReachAvoid(x0,T)           # Exp4
    # scenario = ReachAvoidAdv(x0,T)        # Exp3
    # scenario = FollowTrajectory(x0,T)       # Exp2
    # scenario = FollowTrajectoryAdv(x0,T)  # Exp1

    # Autograd based and Explicit Gradient
    measureType = 1 # 0: Non-Smooth, 1: Standard Smooth, 2: Under Approx, 3: Over Approx, 4: Reversed Standard
    def costFunction(u,measureType=measureType):
        return scenario.costFunction(u,measureType)
        # return (scenario.costFunction(u,2) + scenario.costFunction(u,3))/2

    # costFunctionAutoGrad = grad(costFunction)
    costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()
    # costFunctionExplicitGrad = lambda u_opt_flat, measureType=measureType, T=T: 0.5*(scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), 2).flatten() + scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), 3).flatten())


    ## Set up and solve an optimization problem over u
    u_init = np.zeros((2,T+1)).flatten()   # initial guess
    u_init = np.random.rand(u_init.shape[0])


    u_opt_flat = u_init
    u_opt_mat = u_opt_flat.reshape((2,T+1))

    # scenario.tuneKParameters(u_opt_flat, measureType, 200) # 50
    # scenario.tuneKParameters(u_opt_flat, measureType, 20) # 50


    # initialization
    # measureTypeInit = 1
    # def costFunctionInit(u,measureTypeInit=measureTypeInit):
    #     return scenario.costFunction(u,measureTypeInit)

    # costFunctionExplicitGradInit = lambda u_opt_flat, measureType=measureTypeInit, T=T: scenario.getRobustnessGrad(u_opt_flat.reshape((2,T+1)), measureType).flatten()
    
    # sol = minimize(costFunctionInit, u_init,
    #         jac = costFunctionExplicitGradInit, # this is the function that gives the gradient (now, given by autograd library)
    #         method='SLSQP')
    # u_init = sol.x
    # end initialization


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
    ## End SLSQP method


    # re run
    # u_initnew = sol.x
    # scenario.tuneKParameters(u_initnew, measureType, 20) # 50
    # start_time = time.time()
    # # sol = minimize(costFunction, u_init,
    # #         jac = costFunctionAutoGrad, # this is the function that gives the gradient (now, given by autograd library)
    # #         method='SLSQP')
    # sol = minimize(costFunction, u_initnew,
    #         jac = costFunctionExplicitGrad, # this is the function that gives the gradient (now, given by autograd library)
    #         method='SLSQP')
    # end_time = time.time()
    # executionTime = end_time-start_time;
    # end rerun



    numOfGreedyIter = 0


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

    meanStdRobustness = meanStdRobustness + robustnessCostStd
    meanStdRobustnessTemp = meanStdRobustness/(realization+1)


    errorBand = scenario.getErrorBand(u_opt_mat, measureType)
    print("Error Band: %1.4f, %1.4f in [%1.4f, %1.4f]"%(robustnessCostStd,robustnessCost,robustnessCost+errorBand[0],robustnessCost+errorBand[1]))

    # print("Cost: %1.3f; con.C: %1.3f; rob.C: %1.3f; con.R %1.3f; obs.R %1.3f; Goal.R %1.3f" %(costValue,controlCost,robustnessCost,controlRobustness,obstacleRobustness,goalRobustness))
    
    dataMatrix.append([costValue,controlCost,robustnessCost,executionTime,controlRobustness,obstacleRobustness,goalRobustness,numOfGreedyIter,robustnessCostStd,controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd,robustnessCost+errorBand[0],robustnessCost+errorBand[1]])


    # fig, axes = plt.subplots(1)
    # # Plot the results
    # axes.set_title("SLSQP Method: Time: %1.3f \n Cost %1.3f; con.C %1.3f; rob.C %1.3f; \n Smt: tot.R %1.3f; con.R %1.3f; obs.R %1.3f; goal.R %1.3f \n Std: tot. R %1.3f con.R %1.3f; obs.R %1.3f; goal.R %1.3f" \
    #     %(executionTime, costValue, controlCost, robustnessCost, robustnessCost, controlRobustness, obstacleRobustness, goalRobustness, robustnessCostStd, controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd))       
    # scenario.plotTrajectory(u_opt_mat,axes)
    # plt.show()

    fig1, ax1 = plt.subplots(1)
    ax1.set_title("SA_AG Method: Realization "+str(realization)+"; Robustness "+str(robustnessCostStd))
    scenario.plotTrajectory(u_opt_mat,ax1)
    plt.savefig('Data/Exp4/SA_AGFig'+str(realization)+'.png')



data = np.asarray(dataMatrix)
save('Data/Exp4/SA_AGData.npy', data)

data = load('Data/Exp4/SA_AGData.npy')
# print(data)
## costValue,controlCost,robustnessCost,executionTime,controlRobustness,obstacleRobustness,
## goalRobustness,numOfGreedyIter,robustnessCostStd,controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd,
## robustnessCost+errorBand[0],robustnessCost+errorBand[1]
print(np.mean(data,axis=0))

print(meanStdRobustness/numOfRealizations)

#  SRM1 : 0.409753574
# 'Data/Exp1_k/RA_3iData.npy' : INitialized with SRM1 and further optimized with SRM4 : 0.5060500938070684
# 'Data/Exp1_k/RA_3i2Data.npy' : INitialized with SRM1 and further optimized with SRM3 : 0.5096217657440922
# 'Data/Exp1_k/RA_3i3Data.npy' : INitialized with SRM1 and further optimized with SRM2 : 0.43908089500978703

#  SRM1 : 0.524211335
# 'Data/Exp2_k/RA_3i1Data.npy' : INitialized with SRM1 and further optimized with SRM3 : 0.905302113625889
# 'Data/Exp2_k/RA_3i2Data.npy' : INitialized with SRM1 and further optimized with SRM4 : 0.5985871846608576
# 'Data/Exp2_k/RA_3i3Data.npy' : INitialized with SRM1 and further optimized with SRM2 : 0.5760178184492121

#  SRM1  : 0.263822693
# 'Data/Exp3_k/RA_3i1Data.npy' : INitialized with SRM1 and further optimized with SRM3 : 0.3276643774681392
# 'Data/Exp3_k/RA_3i2Data.npy' : INitialized with SRM1 and further optimized with SRM4 : 0.3163918029311312
# 'Data/Exp3_k/RA_3i3Data.npy' : INitialized with SRM1 and further optimized with SRM2 : 0.22021460554243966


#  SRM1  : 0.181685328
# 'Data/Exp4_k/RA_3i1Data.npy' : INitialized with SRM1 and further optimized with SRM3 : 0.40170469954783705
# 'Data/Exp4_k/RA_3i2Data.npy' : INitialized with SRM1 and further optimized with SRM4 : 0.20269120634101795
# 'Data/Exp4_k/RA_3i3Data.npy' : INitialized with SRM1 and further optimized with SRM2 : 0.33902137440464986
