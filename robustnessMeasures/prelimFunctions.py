import autograd.numpy as np

def getWeightedSum(scalarArray, matrixArray):
    P = len(matrixArray[0][0])
    T = len(matrixArray[0])
    sumVal = np.zeros((T,P))
    for i in range(len(scalarArray)):
        sumVal = sumVal + scalarArray[i]*matrixArray[i]

    return sumVal

def prepend(parameterAddresses,element):
    # print("HHHH")
    # print(parameterAddresses)
    for i in range(len(parameterAddresses)):
        parameterAddresses[i].insert(0,element)
    
    return parameterAddresses













