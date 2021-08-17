# To test grad semantics
import numpy as np

muFun = lambda x ,y: y-2*x+4
muFunGrad = lambda x, y: [-2,1]


rob = lambda s, t, theta : muFun(s[t,0],s[t,1])
robGrad = lambda s, t, theta : [[0 for i in range(len(s[0]))] if j!=t else np.append(muFunGrad(s[1,0],s[1,1]), [0,0]) for j in range(len(s))]

# need to creat an array of arrays that has all zeros, and have same dimentions as s, i.e., (T+1)x4
# only the t^th element should be non zero 


s = [[1,2,4,3],[3,2,4,2],[2,3,2,2],[6,4,3,2],[4,3,2,2],[7,4,5,8]]
print(len(s))
print(len(s[0]))
print([0 for i in range(len(s[0]))])
print([[0 for i in range(len(s[0]))] for i in range(len(s))])
print([[0 for i in range(len(s[0]))] if i!=2 else [5,5,4,5] for i in range(len(s))])
list_comp = [x * 2 if x%2==0 else x/2 for x in range(5)]
print(list_comp)

s = np.asarray(s)
print(s[0,3])
print(np.append(muFunGrad(s[1,0],s[1,1]), [0,0]))
print(robGrad(s,1,0))
print(len(s[0]))


print("Here")
x = np.asarray([2,3,4,2,2,4,5])
print(x)
k = 4

minVal = (-1/float(k)) * np.log(np.sum(np.exp(-k*x)))
minValgrad = np.exp(-k*x)/np.sum(np.exp(-k*x))
print(minVal)
print(minValgrad)
# print(minValgrad*100)

a = [np.zeros((2,5)), np.zeros((2,5))]
print(a[0])
print(a[1]*5+a[0]*7)
print(len(a[0]))

from robustnessMeasures.STLFormulaStandardApprox import STLFormulaSA
STLFormulaSA.getWeightedSum([2,6],a)

print(a[0])
print(a[0].tolist())

print(np.append([2,3],[5,6]).tolist())




print("Here")

# create the lower triangular block matrix with identity matrices I_2
T = 5
I = np.eye(2)
O = np.zeros((2,2))
# block = np.block([[O,O],[I,O]])
# block = np.block([[O,O,O,O],[I,O,O,O],[I,I,O,O],[I,I,I,O]])
block = np.block([[I if t>tau else O for tau in range(T+1)] for t in range(T+1)])
print(block)

print(s)
s_y = s[:,0:2]
s_u = s[:,2:4]
print(s_u)
print(s_y)
print(s_y.flatten())
print(s_y.flatten()@block)
print(np.reshape(s_y.flatten()@block, (T+1,2)))
print(s_u)
print(s_u + np.reshape(s_y.flatten()@block, (T+1,2)))

print(np.asarray([2,3,4]).mean())
# np.mean(np.pows_u-s_y)





