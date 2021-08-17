import autograd.numpy as np

class STLFormulaNS:
    """
    A class for actual non-smooth STL Formulas. 
    """

    def __init__(self, robustness, errorBand, parameters, robustnessGrad=[]):
        """
        An STL Formula is initialized with a robustness function, error band function and parameters of smooth operators
        Arguments:
            robustness : a function that maps from signal s and time t to a scalar value 
            errorBand  : a function that maps from signal s and time t to a two scalar values [L,U] 
            parameters :  
        """

        self.robustness = robustness
        self.errorBand = errorBand
        self.parameters = parameters
        self.robustnessGrad = robustnessGrad
        
        print("An STLFormulaNS object was created")


    def negation(self):
        """
        Return a new STL Formula object which represents the negation
        of this one. The robustness degree is given by
            rho(s,-phi,t) = -rho(s,phi,t)
        """
        newParameters = self.parameters

        newRobustness = lambda s, t, theta : - self.robustness(s,t,theta)
        
        minBand = lambda s, t, theta : - self.errorBand[1](s,t,theta)
        maxBand = lambda s, t, theta : - self.errorBand[0](s,t,theta)
        newErrorBand = [minBand, maxBand]

        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    def conjunction(self, second_formula):
        """
        Return a new STL Formula object which represents the conjuction of
        this formula with second_formula:
            rho(s,phi1^phi2,t) = min( rho(s,phi1,t), rho(s,phi2,t) )
        Arguments:
            second_formula : an STL Formula or predicate defined over the same signal s.
        """
        newParameters = [self.parameters, second_formula.parameters] 

        # new_robustness = lambda s, t : min( self.robustness(s,t), second_formula.robustness(s,t) )
        newRobustness = lambda s, t, theta : STLFormulaNS.minFun([self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])])     
        
        # in the non smooth measure, error band is [min of all the lower bounds, max of all the upper bounds]   
        minBand = lambda s, t, theta : min(self.errorBand[0](s,t,theta[0]), second_formula.errorBand[0](s,t,theta[1]))
        maxBand = lambda s, t, theta : max(self.errorBand[1](s,t,theta[0]), second_formula.errorBand[1](s,t,theta[1]))
        newErrorBand = [minBand, maxBand]
        
        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    def conjunction(formulas):
        # formulas are objects of this class
        L = len(formulas)

        newParameters = [formulas[i].parameters for i in range(L)]
        
        # new_robustness = lambda s, t : min( self.robustness(s,t), second_formula.robustness(s,t) )
        newRobustness = lambda s, t, theta : STLFormulaNS.minFun([formulas[i].robustness(s,t,theta[i]) for i in range(L)])

        # in the non smooth measure, error band is [min of all the lower bounds, max of all the upper bounds]   
        minBand = lambda s, t, theta : min([formulas[i].errorBand[0](s,t,theta[i]) for i in range(L)])             
        maxBand = lambda s, t, theta : max([formulas[i].errorBand[1](s,t,theta[i]) for i in range(L)])             
        newErrorBand = [minBand, maxBand]
        
        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    def disjunction(self, second_formula):
        """
        Return a new STL Formula object which represents the disjunction of
        this formula with second_formula:
            rho(s, phi1 | phi2, t) = max( rho(s,phi1,t), rho(s,phi2,t) )
        Arguments:
            second_formula : an STL Formula or predicate defined over the same signal s.
        """    
        newParameters = [self.parameters, second_formula.parameters] 

        # new_robustness = lambda s, t : max( self.robustness(s,t), second_formula.robustness(s,t) )    
        newRobustness = lambda s, t, theta : STLFormulaNS.maxFun([self.robustness(s,t,theta[0]), second_formula.robustness(s,t,theta[1])])        
        
        # in the non smooth measure, error band is [min of all the lower bounds, max of all the upper bounds]   
        minBand = lambda s, t, theta : min(self.errorBand[0](s,t,theta[0]), second_formula.errorBand[0](s,t,theta[1]))
        maxBand = lambda s, t, theta : max(self.errorBand[1](s,t,theta[0]), second_formula.errorBand[1](s,t,theta[1]))
        newErrorBand = [minBand, maxBand]

        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    def disjunction(formulas):
        # formulas are objects of this class
        L = len(formulas)

        newParameters = [formulas[i].parameters for i in range(L)]
        
        # new_robustness = lambda s, t : max( self.robustness(s,t), second_formula.robustness(s,t) )
        newRobustness = lambda s, t, theta : STLFormulaNS.maxFun([formulas[i].robustness(s,t,theta[i]) for i in range(L)]) 

        # in the non smooth measure, error band is [min of all the lower bounds, max of all the upper bounds] 
        minBand = lambda s, t, theta : min([formulas[i].errorBand[0](s,t,theta[i]) for i in range(L)])             
        maxBand = lambda s, t, theta : max([formulas[i].errorBand[1](s,t,theta[i]) for i in range(L)])             
        newErrorBand = [minBand, maxBand]

        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    def eventually(self, t1, t2):
        """
        Return a new STL Formula object which represents this formula holding
        at some point in [t+t1, t+t2].
            rho(s, F_[t1,t2](phi), t) = max_{k in [t+t1,t+t2]}( rho(s,phi,k) )
        Arguments:
            t1 : an integer between 0 and signal length T
            t2 : an integer between t1 and signal length T
        """
        L = t2 + 1 - t1 

        newParameters = [self.parameters]

        # new_robustness = lambda s, t : max([ self.robustness(s,k) for k in range(t+t1, t+t2+1)])
        newRobustness = lambda s, t, theta : STLFormulaNS.maxFun([self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])        
        
        # in the non smooth measure, error band is [min of all the lower bounds, max of all the upper bounds] 
        minBand = lambda s, t, theta : min([self.errorBand[0](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])        
        maxBand = lambda s, t, theta : max([self.errorBand[1](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])        
        newErrorBand = [minBand, maxBand]

        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    def always(self, t1, t2):
        """
        Return a new STL Formula object which represents this formula holding
        at all times in [t+t1, t+t2].
            rho(s, F_[t1,t2](phi), t) = min_{k in [t+t1,t+t2]}( rho(s,phi,k) )
        Arguments:
            t1 : an integer between 0 and signal length T
            t2 : an integer between t1 and signal length T
        """
        L = t2 + 1 - t1 

        newParameters = [self.parameters]

        # new_robustness = lambda s, t : min([ self.robustness(s,k) for k in range(t+t1, t+t2+1)])
        newRobustness = lambda s, t, theta : STLFormulaNS.minFun([self.robustness(s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])        
        
        # in the non smooth measure, error band is [min of all the lower bounds, max of all the upper bounds] 
        minBand = lambda s, t, theta : min([self.errorBand[0](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])        
        maxBand = lambda s, t, theta : max([self.errorBand[1](s, tau, theta[0]) for tau in range(t+t1, t+t2+1)])        
        newErrorBand = [minBand, maxBand]

        return STLFormulaNS(newRobustness, newErrorBand, newParameters)


    
    def minFun(values, k=0):
        return min(values)



    def maxFun(values, k=0):
        return max(values)

        