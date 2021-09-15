from numpy import linalg as LA 
from numpy import save
from numpy import load
import numpy as np
import matplotlib.pyplot as plt


# prefix = 'Data/Exp4'
# prefix = 'Data/Exp1_k'
prefix = 'Data/Exp4'

# data = load(prefix+'/SAData.npy')
# data = load(prefix+'/SA_9Data.npy')
data = load(prefix+'/SA_AGData.npy')
# print(repr(data))
print(*np.mean(data,axis=0))


# data = load(prefix+'/UAData.npy')
# data = load(prefix+'/UA_9Data.npy')
data = load(prefix+'/UA_AGData.npy')
# print(repr(data))
print(*np.mean(data,axis=0))


# data = load(prefix+'/OAData.npy')
# data = load(prefix+'/OA_9Data.npy')
# data = load(prefix+'/OA_AGData.npy')
# print(repr(data))
# print(*np.mean(data,axis=0))


# data = load(prefix+'/RAData.npy')
# data = load(prefix+'/RA_9Data.npy')
# data = load(prefix+'/OA_AGData.npy')
# print(repr(data))
# print(*np.mean(data,axis=0))


## costValue,controlCost,robustnessCost,executionTime,controlRobustness,obstacleRobustness,
## goalRobustness,numOfGreedyIter,robustnessCostStd,controlRobustnessStd,obstacleRobustnessStd,goalRobustnessStd,
## robustnessCost+errorBand[0],robustnessCost+errorBand[1]
# print(np.mean(data[:,2]),np.mean(data[:,8]))
print(np.mean(data[:,2]),np.mean(data[:,8]))

# errors = data[:,8]-data[:,2]
# errorsL = data[:,12]-data[:,2] 
# errorsH = data[:,13]-data[:,2] 
# errorsL = np.min(errorsL)
# errorsH = np.max(errorsH)
# print(errorsL,errorsH)
# bins = np.arange(errorsL,errorsH,0.1) 
# print(errors)
# plt.hist(errors, bins=bins)
# plt.title("Histogram with 'auto' bins")
# plt.show()