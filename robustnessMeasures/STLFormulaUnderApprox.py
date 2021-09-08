import autograd.numpy as np
from robustnessMeasures.prelimFunctions import *

class STLFormulaUA:
    """
    A class for standard smooth STL Formulas (using LSE based smooth-min and smooth-max operators). 
    """

    def __init__(self, robustness, errorBand, parameters, paraAddresses = [], paraTypes=[], robustnessGrad=[]):
        """
        An STL Formula is initialized with a robustness function, error band function and parameters of smooth operators
        Arguments:
            robustness : a function that maps from signal s, time t, parameters theta, to a scalar value 
            errorBand  : a function that maps from signal s , time t, parameters theta, to two scalar values [L,U] 
            parameters : default set of parameter values to use as the parameters theta
        """

        self.robustness = robustness # a function of 
        self.errorBand = errorBand
        self.parameters = parameters
        self.robustnessGrad = robustnessGrad

        # self.minOperatorBounds[0] = lambda k, m: 0
        # self.minOperatorBounds[1] = lambda k, m: np.log(m)/k

        # self.maxOperatorBounds[0] = lambda k, m: -np.log(m)/k
        # self.maxOperatorBounds[1] = lambda k, m: 0

        self.minOperatorK = 3.0
        self.maxOperatorK = 3.0

        self.paraAddresses = paraAddresses # where to find tunable parameters (minOperatorK,maxOperatork): list of lists
        self.paraTypes = paraTypes     # what type is each parameter (min or max, with how many parameters)
        
        # print("An STLFormulaUA object was created")


    def smoothMinMaxOperatorBounds(minmax,lowup,x=[0,0,1]):
        if minmax == 0 and lowup == 0:      # min operator lower bound
            return lambda k, m: 0
        elif minmax == 0 and lowup == 1:    # min operator upper bound
            return lambda k, m: np.log(m)/k   ## this is what makes the error bound to span downward
            # return lambda k, m: 0
        elif minmax == 1 and lowup == 0:    # max operator lower bound
            return lambda k, m: 0 
        elif minmax == 1 and lowup == 1:    # max operator upper bound
            a_1 = max(x)
            a_m = min(x)

            # x.pop(x.index(a_1))
            # a_2 = max(x)
            # return lambda k, m, a_1=a_1, a_2=a_2, a_m=a_m: (a_1-a_m)/((np.exp(k*(a_1-a_2))/(m-1))+1) 
            
            # x = np.array(x) # the following bound is more tighter!
            # return lambda k, m, a_1=a_1, a_m=a_m, x=x: (a_1-a_m)*(np.sum(np.exp(k*x)) - np.exp(k*a_1))/(np.sum(np.exp(k*x)))
            
            xa_1 = np.array(x)-a_1 # the following bound is more tighter!
            # print(xa_1)
            # print((1 - 1/(np.sum(np.exp(50*xa_1)))))
            return lambda k, m, a_1=a_1, a_m=a_m, x=x: (a_1-a_m)*(1 - 1/(np.sum(np.exp(k*xa_1))))


    def negation(self):
        """
        Return a new STL Formula object which represents the negation
        of this one. The robustness degree is given by
            rho(s,-phi,t) = -rho(s,phi,t)
        """        
        newParameters = self.parameters
        newParaAddresses = self.paraAddresses
        newParaTypes = self.paraTypes

        newRobustness = lambda s, t, theta : -self.robustness(s,t,theta)
        
        minBand = lambda s, t, theta : - self.errorBand[1](s,t,theta)
        maxBand = lambda s, t, theta : - self.errorBand[0](s,t,theta)
        newErrorBand = [minBand, maxBand]

        newRobustnessGrad = lambda s, t, theta : - self.robustnessGrad(s,t,theta)
        
        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)


    def conjunction(self, second_formula, k=-1):
        """
        Return a new STL Formula object which represents the conjuction of
        this formula with second_formula:
            rho(s,phi1^phi2,t) = min( rho(s,phi1,t), rho(s,phi2,t) )
        Arguments:
            second_formula : an STL Formula or predicate defined over the same signal s.
        """
        
        # new default parameters
        if k == -1:
            k = self.minOperatorK

        newParameters = [self.parameters, second_formula.parameters, k] 
        newParaAddresses = prepend(self.paraAddresses,0) + prepend(second_formula.paraAddresses,1)+[[2]]
        newParaTypes = self.paraTypes + second_formula.paraTypes + [-2] #- signals that this is a min operator, magnitude indicates the num of elements 

        # new_robustness = lambda s, t : self.min_test( [self.robustness(s,t),second_formula.robustness(s,t)])
        newRobustness = lambda s, t, theta : STLFormulaUA.minFun([self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])], theta[2])        
        
        # in the smooth std approx measure, error band is:
        # [L_{k_1,2}^{min} + min{all the lower bounds},    U_{k_1,2}^{min} + max{all the upper bounds}]   
        minBand = lambda s, t, theta: min(self.errorBand[0](s,t,theta[0]), second_formula.errorBand[0](s,t,theta[1])) + STLFormulaUA.smoothMinMaxOperatorBounds(0,0,[self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])])(theta[2],2)
        maxBand = lambda s, t, theta: max(self.errorBand[1](s,t,theta[0]), second_formula.errorBand[1](s,t,theta[1])) + STLFormulaUA.smoothMinMaxOperatorBounds(0,1,[self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])])(theta[2],2)
        newErrorBand = [minBand, maxBand]
               
        newRobustnessGrad = lambda s, t, theta : getWeightedSum(STLFormulaUA.minFunGrad([self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])], theta[2]), [self.robustnessGrad(s,t,theta[0]), second_formula.robustnessGrad(s,t,theta[1])])   

        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)


    def conjunction(formulas, k=-1):
        # formulas are objects of this class
        L = len(formulas)

        # new default parameters
        if k==-1:
            k = formulas[0].minOperatorK

        newParameters = [formulas[i].parameters for i in range(L)]+[k]
        newParaAddresses = sum([prepend(formulas[i].paraAddresses,i) for i in range(L)],[])+[[L]]
        newParaTypes = sum([formulas[i].paraTypes for i in range(L)],[])+[-L]

        # new_robustness = lambda s, t : min( self.robustness(s,t), second_formula.robustness(s,t) )
        newRobustness = lambda s, t, theta : STLFormulaUA.minFun([formulas[i].robustness(s,t,theta[i]) for i in range(L)], theta[L])   
        
        # in the smooth std approx measure, error band is:
        # [L_{k_1,L}^{min} + min{all the lower bounds},    U_{k_1,L}^{min} + max{all the upper bounds}]  
        minBand = lambda s, t, theta : min([formulas[i].errorBand[0](s,t,theta[i]) for i in range(L)]) + STLFormulaUA.smoothMinMaxOperatorBounds(0,0,[formulas[i].robustness(s,t,theta[i]) for i in range(L)])(theta[L],L)   
        maxBand = lambda s, t, theta : max([formulas[i].errorBand[1](s,t,theta[i]) for i in range(L)]) + STLFormulaUA.smoothMinMaxOperatorBounds(0,1,[formulas[i].robustness(s,t,theta[i]) for i in range(L)])(theta[L],L)             
        newErrorBand = [minBand, maxBand]        

        newRobustnessGrad = lambda s, t, theta : getWeightedSum(STLFormulaUA.minFunGrad([formulas[i].robustness(s,t,theta[i]) for i in range(L)], theta[L]), [formulas[i].robustnessGrad(s,t,theta[i]) for i in range(L)])   

        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)


    def disjunction(self, second_formula, k=-1): 
        """
        Return a new STL Formula object which represents the disjunction of
        this formula with second_formula:
            rho(s, phi1 | phi2, t) = max( rho(s,phi1,t), rho(s,phi2,t) )
        Arguments:
            second_formula : an STL Formula or predicate defined over the same signal s.
        """

        # new default parameters
        if k == -1:
            k = self.maxOperatorK

        newParameters = [self.parameters, second_formula.parameters, k]
        newParaAddresses = prepend(self.paraAddresses,0) + prepend(second_formula.paraAddresses,1)+[[2]]
        newParaTypes = self.paraTypes + second_formula.paraTypes + [2] #- signals that this is a min operator, magnitude indicates the num of elements 

        # newRobustness = lambda s, t : STLFormulaUA.maxFun(self.robustness(s,t), second_formula.robustness(s,t))        
        newRobustness = lambda s, t, theta : STLFormulaUA.maxFun([self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])], theta[2])        
        
        # in the smooth std approx measure, error band is:
        # [L_{k_1,2}^{max} + min{all the lower bounds},    U_{k_1,2}^{max} + max{all the upper bounds}]   
        minBand = lambda s, t, theta: min(self.errorBand[0](s,t,theta[0]), second_formula.errorBand[0](s,t,theta[1])) + STLFormulaUA.smoothMinMaxOperatorBounds(1,0,[self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])])(theta[2],2)
        maxBand = lambda s, t, theta: max(self.errorBand[1](s,t,theta[0]), second_formula.errorBand[1](s,t,theta[1])) + STLFormulaUA.smoothMinMaxOperatorBounds(1,1,[self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])])(theta[2],2)
        newErrorBand = [minBand, maxBand]

        newRobustnessGrad = lambda s, t, theta : getWeightedSum(STLFormulaUA.maxFunGrad([self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])], theta[2]), [self.robustnessGrad(s,t,theta[0]), second_formula.robustnessGrad(s,t,theta[1])])   

        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)


    def disjunction(formulas, k=-1):
        # formulas are objects of this class
        L = len(formulas)

        # new default parameters
        if k==-1:
            k = formulas[0].maxOperatorK

        newParameters = [formulas[i].parameters for i in range(L)]+[k]
        newParaAddresses = sum([prepend(formulas[i].paraAddresses,i) for i in range(L)],[])+[[L]]
        newParaTypes = sum([formulas[i].paraTypes for i in range(L)],[])+[L]

        # newRobustness = lambda s, t : STLFormulaUA.maxFun([formulas[i].robustness(s,t) for i in range(len(formulas))])        
        newRobustness = lambda s, t, theta : STLFormulaUA.maxFun([formulas[i].robustness(s,t,theta[i]) for i in range(L)], theta[L])   

        # in the smooth std approx measure, error band is:
        # [L_{k_1,L}^{max} + min{all the lower bounds},    U_{k_1,L}^{max} + max{all the upper bounds}]  
        minBand = lambda s, t, theta : min([formulas[i].errorBand[0](s,t,theta[i]) for i in range(L)]) + STLFormulaUA.smoothMinMaxOperatorBounds(1,0,[formulas[i].robustness(s,t,theta[i]) for i in range(L)])(theta[L],L)          
        maxBand = lambda s, t, theta : max([formulas[i].errorBand[1](s,t,theta[i]) for i in range(L)]) + STLFormulaUA.smoothMinMaxOperatorBounds(1,1,[formulas[i].robustness(s,t,theta[i]) for i in range(L)])(theta[L],L)             
        newErrorBand = [minBand, maxBand] 

        newRobustnessGrad = lambda s, t, theta : getWeightedSum(STLFormulaUA.maxFunGrad([formulas[i].robustness(s,t,theta[i]) for i in range(L)], theta[L]), [formulas[i].robustnessGrad(s,t,theta[i]) for i in range(L)])   

        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)


    def eventually(self, t1, t2, k=-1):
        """
        Return a new STL Formula object which represents this formula holding
        at some point in [t+t1, t+t2].
            rho(s, F_[t1,t2](phi), t) = max_{k in [t+t1,t+t2]}( rho(s,phi,k) )
        Arguments:
            t1 : an integer between 0 and signal length T
            t2 : an integer between t1 and signal length T
        """
        L = t2 + 1 - t1 
        
        # new parameters
        if k==-1:
            k = self.maxOperatorK

        newParameters = [self.parameters, k]
        newParaAddresses = prepend(self.paraAddresses,0)+[[1]]
        newParaTypes = self.paraTypes + [1]

        # new_robustness = lambda s, t : self.max_test([ self.robustness(s,k) for k in range(t+t1, t+t2+1)])
        newRobustness = lambda s, t, theta : STLFormulaUA.maxFun([self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)], theta[1])        
        
        # in the smooth std approx measure, error band is:
        # [L_{k_1,L}^{max} + min{all the lower bounds},    U_{k_1,L}^{max} + max{all the upper bounds}]
        minBand = lambda s, t, theta : min([self.errorBand[0](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)]) + STLFormulaUA.smoothMinMaxOperatorBounds(1,0,[self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])(theta[1],L)       
        maxBand = lambda s, t, theta : max([self.errorBand[1](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)]) + STLFormulaUA.smoothMinMaxOperatorBounds(1,1,[self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])(theta[1],L)       
        newErrorBand = [minBand, maxBand]

        newRobustnessGrad = lambda s, t, theta : getWeightedSum(STLFormulaUA.maxFunGrad([self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)], theta[1]), [self.robustnessGrad(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])   

        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)

    def always(self, t1, t2, k=-1):
        """
        Return a new STL Formula object which represents this formula holding
        at all times in [t+t1, t+t2].
            rho(s, F_[t1,t2](phi), t) = min_{k in [t+t1,t+t2]}( rho(s,phi,k) )
        Arguments:
            t1 : an integer between 0 and signal length T
            t2 : an integer between t1 and signal length T
        """
        L = t2 + 1 - t1

        # new parameters
        if k==-1:
            k = self.minOperatorK

        newParameters = [self.parameters, k]
        newParaAddresses = prepend(self.paraAddresses,0)+[[1]]
        newParaTypes = self.paraTypes + [-1]

        # newRobustness = lambda s, t : STLFormulaUA.minFun([self.robustness(s,tau) for tau in range(t+t1, t+t2+1)])        
        newRobustness = lambda s, t, theta : STLFormulaUA.minFun([self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)], theta[1])        

        # in the smooth std approx measure, error band is:
        # [L_{k_1,L}^{min} + min{all the lower bounds},    U_{k_1,L}^{min} + max{all the upper bounds}]
        minBand = lambda s, t, theta : min([self.errorBand[0](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)]) + STLFormulaUA.smoothMinMaxOperatorBounds(0,0,[self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])(theta[1],L)       
        maxBand = lambda s, t, theta : max([self.errorBand[1](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)]) + STLFormulaUA.smoothMinMaxOperatorBounds(0,1,[self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])(theta[1],L)       
        newErrorBand = [minBand, maxBand]

        newRobustnessGrad = lambda s, t, theta : getWeightedSum(STLFormulaUA.minFunGrad([self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)], theta[1]), [self.robustnessGrad(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])   

        return STLFormulaUA(newRobustness, newErrorBand, newParameters, newParaAddresses, newParaTypes, newRobustnessGrad)
        
    
    def minFun(x, theta):
        """
        Compute the approximate minimum value of a list. 
        """
        # k = len(x)
        k = theta

        x = np.array(x)
        x_min = np.min(x)
        return x_min-(1/k)*np.log(np.sum(np.exp(-k*(x-x_min))))

    def maxFun(x, theta):
        """
        Compute the approximate maximum value of a list. 
        """
        # a = 2*np.log(len(x))
        k = theta

        x = np.array(x)
        x_max = np.max(x)
        exp = np.exp(k*(x-x_max))
        return np.sum(x*exp)/np.sum(exp)

    def minFunGrad(x, theta):
        """
        Compute the gradient components (i.e., partial minFun / partial x), as an array (of length same as x)
        """
        k = theta

        x = np.array(x)
        x_min = np.min(x)
        return (1/np.sum(np.exp(-k*(x-x_min))))*np.exp(-k*(x-x_min))

    def maxFunGrad(x, theta):
        """
        Compute the gradient components (i.e., partial maxFun / partial x), as an array (of length same as x)
        """
        k = theta

        x = np.array(x)
        x_max = np.max(x)
        return (1/np.sum(np.exp(k*(x-x_max))))*np.exp(k*(x-x_max))*(1-k*(STLFormulaUA.maxFun(x,theta)-x))

