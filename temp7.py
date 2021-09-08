import time
from numpy import linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad
# import autograd.numpy as np

# The detailed implementation of this scenario is defined here:
from scenarios import ReachAvoid 
from scenarios import ReachAvoidAdv 

from simulations import DataCollector

# initialize the example with an initial state
T = 20   #20 number of timesteps
x0 = np.asarray([0,0])[:,np.newaxis] # [0.5,5] for ReachAvoid4
scenario = ReachAvoid(x0,T)
scenario = ReachAvoidAdv(x0,T)



# # Set up and solve an optimization problem over u
np.random.seed(7)
u_test = np.zeros((2,T+1)).flatten()   # initial guess
u_test = np.random.rand(u_test.shape[0])

measureType = 2

costFunctionGrad = grad(scenario.costFunction)
gradVal = costFunctionGrad(u_test, measureType)

print("Cost, Robustness, ErrorBand:")
costValue = scenario.costFunction(u_test, measureType)
u_test = u_test.reshape((2,T+1))
robustness = scenario.getRobustness(u_test, measureType)
errorBand = scenario.getErrorBand(u_test, measureType)
print(costValue, robustness, errorBand)

###  update k

# get signal 
signal = scenario.getSignal(u_test)
measureTypeIndex = scenario.requiredMeasureTypes.index(measureType)

# def errorBandWidth([k_1,k_2]):



# grad(scenario.costFunction)
spec = scenario.fullSpec.STLFormulaObjects[measureTypeIndex]
print("Parameters: ")
print(spec.parameters)

print("ParaAddresses: ")
print(spec.paraAddresses)

print("ParaTypes: ")
print(spec.paraTypes)

def getVal(valueArray,addressArray):
    val = valueArray
    addressStr = ''
    for i in addressArray:
        addressStr = addressStr + '['+str(i)+']'
        val = val[i]
    return val

def setVal(valueArray,addressArray,increment):
    
    val = valueArray
    addressStr = ''
    for i in addressArray:
        addressStr = addressStr + '['+str(i)+']'
        val = val[i]
    val = val + increment

    exec('valueArray'+addressStr+'=val',locals(),globals())
    return valueArray



def getErrorBand(kValueArray,measureType=measureType):
    return scenario.getErrorBand(u,measureType,kValueArray)    
    # set k values to all the array

    




ErrorL = scenario.fullSpec.STLFormulaObjects[measureTypeIndex].errorBand[0](signal.T, 0, spec.parameters)
ErrorH = scenario.fullSpec.STLFormulaObjects[measureTypeIndex].errorBand[1](signal.T, 0, spec.parameters)

print("Error Band")
print(ErrorL, ErrorH)

for i in range(len(spec.paraTypes)):
    if spec.paraTypes[i]!=0:
        setVal(spec.parameters, spec.paraAddresses[i],0)
        # print("i",i)


ErrorL = scenario.fullSpec.STLFormulaObjects[measureTypeIndex].errorBand[0](signal.T, 0, spec.parameters)
ErrorH = scenario.fullSpec.STLFormulaObjects[measureTypeIndex].errorBand[1](signal.T, 0, spec.parameters)

print("Error Band")
print(ErrorL, ErrorH)


# create a function bandwidth(kArray)

# spec.paraTypes2 = [spec.paraTypes[i] for i in range(len(spec.paraTypes)) if spec.paraTypes[i]!=0]
# spec.paraAddresses2 = [spec.paraAddresses[i] for i in range(len(spec.paraTypes)) if spec.paraTypes[i]!=0]
# print(spec.paraTypes2)
# print(spec.paraAddresses2)
# print(spec.parameters)


def getValue(valArray,indArray):
    if len(indArray)==1:
        return valArray[indArray[0]]
    else:
        i = indArray[0]
        return getValue(valArray[i],indArray[1:])

def setValue(valArray,indArray,newVal,valArrayComplete):
    if len(indArray)==1:
        return [valArray[indArray[0]], valArrayComplete]
    else:
        i = indArray[0]
        return getValue(valArray[i],indArray[1:],valArrayComplete)

print("Parameters: ")
print(spec.parameters)

print("ParaAddresses: ")
print(spec.paraAddresses)


print(getValue(spec.parameters,spec.paraAddresses[2]))

# def giveBandWidth(kArray):


errorBand = scenario.getErrorBand(u_test, measureType)
print(costValue, robustness, errorBand)



kValueArray = [] 
for address in spec.paraAddresses:
    parameters = spec.parameters
    kVal = DataCollector.getValue(parameters,address)
    kValueArray.append(kVal)

kValueArray = np.asarray(kValueArray)
print(kValueArray)
print(scenario.getErrorBandWidth(u_test, measureType, []))
print(scenario.getErrorBandWidth(u_test, measureType, kValueArray))
print(scenario.getErrorBandWidth(u_test, measureType, []))



# # Set up and solve an optimization problem over u

costFunctionGrad = grad(scenario.costFunction)

# u_test = u_test.flatten()

def costFunction(kValueArray, u=u_test, measureType=measureType):
    return scenario.getErrorBandWidth(u, measureType, kValueArray)

costFunctionAutoGrad = grad(costFunction)

precision = 0.01
kMax = 10.0
kMin = 0.1

kInit = kValueArray
initBand = scenario.getErrorBand(u_test,measureType)
initWidth = scenario.getErrorBandWidth(u_test, measureType, kValueArray)
oldWidth = initWidth
for i in range(200):
    gradVal = costFunctionAutoGrad(kValueArray,u_test,measureType)
    newkValueArray = kValueArray - 2*gradVal
    newkValueArray[newkValueArray>kMax] = kMax
    newkValueArray[newkValueArray<kMin] = kMin
    kValueArray = newkValueArray

    width = scenario.getErrorBandWidth(u_test, measureType, kValueArray)
    normGrad = LA.norm(gradVal)
    widthImprovement = oldWidth - width
    oldWidth = width
    print("i = "+str(i)+"; errot band width = "+"{:.3f}".format(width)+"; normGrad="+"{:.4f}".format(normGrad)+"; widthImpr.="+"{:.4f}".format(widthImprovement))
    print("k="+"".join(["{:.3f}".format(kValueArray[j])+", " for j in range(len(kValueArray))]))


    if widthImprovement<precision and normGrad<precision:
        # print(gradVal)

        print("Local optimum reached at iteration %i" %i)
        break
        
print("Old and new k values: ")
print("k="+"".join(["{:.3f}".format(kInit[j])+", " for j in range(len(kInit))]))
print("k="+"".join(["{:.3f}".format(kValueArray[j])+", " for j in range(len(kValueArray))]))
print("Old and new errro bands: ")
print(initBand)
print(scenario.getErrorBand(u_test,measureType))
print("Total Improvement="+str(initWidth-width))

# # grad(scenario.costFunction)
spec = scenario.fullSpec.STLFormulaObjects[measureTypeIndex]
# print("Parameters: ")
print(spec.parameters)
print(spec.paraAddresses)


