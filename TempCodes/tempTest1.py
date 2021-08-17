# To test error band parameters concatenation

# Spec 1
rob_1 = lambda s, t, theta_1 : s[t]     # robustness  
err_1 = lambda s, t, theta_1 : theta_1	# lowerbound of the error bound of the robusteness
para_1 = 0.5 							# default value of [theta_1]

# Spec 2
rob_2 = lambda s, t, theta_2 : 2*s[t]	# robustness  
err_2 = lambda s, t, theta_2 : theta_2	# lowerbound of the error bound of the robusteness
para_2 = 0.5 							# default value of [theta_2]


def minFun(a,b,k):
	return (a+b)/k

def exactMin(a,b):
	return (a+b)/2

def minFunError(k,m):
	return k/m



# combined: Spec1 AND Spec 2 ===> Spec 3
k = 2.0
para_3 = [para_1, para_2, k] # default value of [theta_3]
rob_3 = lambda s, t, theta_3: minFun( rob_1(s,t,theta_3[0]), rob_2(s,t,theta_3[1]), theta_3[2])  
err_3 = lambda s, t, theta_3: exactMin( err_1(s,t,theta_3[0]), err_2(s,t,theta_3[1])) + minFunError(theta_3[2],2) 

s = [1,2,3,4,5]
t = 2
rob_3(s,t,para_3)
err_3(s,t,para_3)

print(rob_3(s,t,para_3))
print(err_3(s,t,para_3))



# Spec 4
rob_4 = lambda s, t, theta_1 : s[t]     # robustness  
err_4 = lambda s, t, theta_1 : theta_1	# lowerbound of the error bound of the robusteness
para_4 = 0.5 							# default value of [theta_4]



# Combined2: Spec3 AND Spec 4 ===> Spec 5
k = 3.0
para_5 = [para_3, para_4, k] # default value of [theta_3]
rob_5 = lambda s, t, theta_5: minFun( rob_3(s,t,theta_5[0]), rob_4(s,t,theta_5[1]), theta_5[2] )  
err_5 = lambda s, t, theta_5: exactMin( err_3(s,t,theta_5[0]), err_4(s,t,theta_5[1]) ) + minFunError(theta_5[2],2) 

s = [1,2,3,4,5]
t = 2

rob_5(s,t,para_5)
err_5(s,t,para_5)

print(rob_5(s,t,para_5))
print(err_5(s,t,para_5))



# Combined2: Spec3 AND Spec 5 ===> Spec 6
k = 4.0
para_6 = [para_3, para_5, k] # default value of [theta_3]
rob_6 = lambda s, t, theta_6: minFun( rob_3(s,t,theta_6[0]), rob_5(s,t,theta_6[1]), theta_6[2] )  
err_6 = lambda s, t, theta_6: exactMin( err_3(s,t,theta_6[0]), err_5(s,t,theta_6[1]) ) + minFunError(theta_6[2],2) 

s = [1,2,3,4,5]
t = 2

print(para_6)
rob_6(s,t,para_6)
err_6(s,t,para_6)

print(rob_6(s,t,para_6))
print(err_6(s,t,para_6))


def funVal(a,k=a[0])
	print(a)
	print(k)

funVal([2,3])
funVal([2,3],5)