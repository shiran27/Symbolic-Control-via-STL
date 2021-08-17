##
#
# This file contains classes that define some simple examples of scenarios
# Each example scenario has can have multiple robustness measures (smooth and non-smooth) (i.e., functions of the signal) 
# Also each each example scenario can have other functions that are recursively defined like the ones for gradients approximation error bounds 
# 
##

import time
from copy import copy
import numpy as np
# import autograd.numpy as np

from matplotlib.patches import Rectangle


from regions import PolyRegion
from STLMainClass import STLFormulas


class ReachAvoid:
    
    """
    This example involves moving a robot with single integrator dynamics past an obstacle and to a goal postion with bounded control effort. 
    It also serves as a template class for more complex examples.
    """
    
    def __init__(self, initial_state, T=20):
        """
        Set up the example scenario with the initial state, which should be a (4,1) numpy
        array with [x,x',y,y'].
        """
        
        self.x0 = np.asarray(initial_state)
        self.T = T  # The time bound of our specification


        # vertices are given in the counterclockwise order, nx2 list, polygon should be convex
        obstacle = PolyRegion(1,'Obstacle',[[3,4],[5,4],[5,6],[3,6]],'red',0.5)
        goal = PolyRegion(2,'Goal',[[7,8],[8,8],[8,9],[7,9]],'green',0.5)
        aux = PolyRegion(3,'Aux',[[7,1],[8,1],[8.5,2],[7.5,3],[6.5,2]])

        uMin, uMax = -1.0, 1.0 
        controlBounds = PolyRegion(4,'Control',[[uMin,uMin],[uMax,uMin],[uMax,uMax],[uMin,uMax]],'green',0.5)
                

        requiredMeasureTypes = [0,1,2,3,4] # 0: absolut, 1:Std smooth, 2:Under Approx, 3:Over Approx, 4: Reveresd
        
        # Now we'll define the STL specification. We'll do this over
        # the signal s, which is a list of x, y coordinates and the controls u_x, u_y
        # input u at each timestep. 

        # Negation can only be used with predicates, because, specs must be in "Disjunctive normal form"

        ## Avoiding the obstacle
        missObstacle = obstacle.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        self.avoidedObstacle = STLFormulas.disjunction(missObstacle, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection
        # self.avoidedObstacle = STLFormulas.conjunction(hitObstacle, requiredMeasureTypes).negation() # we get one STLFormula collection

        # ## Reaching the Goal
        reachedGoal = goal.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        self.reachedGoal = STLFormulas.conjunction(reachedGoal, requiredMeasureTypes).eventually(0, self.T) # we get one STLFormula collection

        ## Bounding the control
        boundedControl = controlBounds.inControlRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        self.boundedControl = STLFormulas.conjunction(boundedControl, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection

        ## Full specification
        fullSpec = [self.avoidedObstacle, self.reachedGoal, self.boundedControl]
        self.fullSpec = STLFormulas.conjunction(fullSpec, requiredMeasureTypes)


        self.transferMatrixYtoU = ReachAvoid.getTransferMatrixYtoU(T)

        # self.regions = [obstacle, goal, aux]
        self.regions = [obstacle, goal]
        self.controlBounds = controlBounds
        self.requiredMeasureTypes = requiredMeasureTypes


    
    def draw(self, ax):
        """
        Create a plot of the obstacle and goal regions on
        the given matplotlib axis.
        """
        ax.set_xlim((0,10))  #(0,12)
        ax.set_ylim((0,10))

        for poly in self.regions:
            poly.draw(ax)

    
    def plotControlProfile(self, u, ax, label=None):
        ax.set_xlim((-2,2)) 
        ax.set_ylim((-2,2))
        self.controlBounds.draw(ax)

        ax.plot(u[0,:], u[1,:], label=label, linestyle="-", marker=".")
        ax.grid()
        ax.set_xlabel("u_x")
        ax.set_ylabel("u_y")


    def plotTrajectory(self, u, ax, label=None):
        """
        Create a plot of the position resulting from applying 
        control u on the matplotlib axis ax. 
        """
        # Plot the goal and obstacle regions
        self.draw(ax)

        # Get the resulting trajectory
        s = self.getSignal(u)
        x = s[0,:]
        y = s[1,:]

        ax.plot(x, y, label=label, linestyle="-", marker=".")


    def getSignal(self, u):
        """ 
        Maps a control signal u and an initial condition to a signal we can check robustness, cost, error bounds, etc. 
        Composition of the signal we consider: (x,y) position of the robot and (u_x,u_yy) components of the control input.
        Arguments:
            u   : a (2,T) numpy array representing the control sequence
        Returns:
            s   : a (4,T) numpy array representing the signal we'll check
        """

        # System definition: x_{t+1} = A*x_t + B*u_t
        A = np.eye(2)    # single integrator
        B = np.eye(2)

        # number of timesteps
        T = u.shape[1]      

        # Starting state
        x = copy(self.x0)

        # Signal that we'll check consists of both states and control inputs 
        s = np.hstack([x.flatten(),u[:,0]])[np.newaxis].T

        # Run the controls through the system and see what we get
        for t in range(1,T):
            # Autograd doesn't support array assignment, so we can't pre-allocate s here
            s_t = np.hstack([x.flatten(),u[:,t]])[np.newaxis].T
            s = np.hstack([s,s_t])

            # Update the system state
            x = A@x + B@u[:,t][:,np.newaxis]   # ensure u is of shape (2,1) before applying

        return s


    def getRobustness(self, controlInput, measureType, spec=None):
        """
        For a given initial state and control input sequence, calculates rho,
        a scalar value which indicates the degree of satisfaction of the specification.
        Arguments:
            controlInput    : a (2,T) numpy array representing the control sequence
            spec : an STLFormula to evaluate (the full specification by default), given as a string: "avoidedObstacle", "reachedGoal" or "boundedControl"
        Returns:
            rho  : a scalar value indicating the degree of satisfaction. Positive values
                     indicate that the specification is satisfied.
        """
        # By default, evaluate the full specification. Otherwise you could pass it 
        # a different formula, such as the (sub)specification for obstacle avoidance.

        signal = self.getSignal(controlInput)

        measureTypeIndex = self.requiredMeasureTypes.index(measureType)
        if spec is None:
            spec = self.fullSpec.STLFormulaObjects[measureTypeIndex]
        else:
            exec("specTemp = self."+spec+".STLFormulaObjects[measureTypeIndex]",locals(),globals())
            spec = specTemp

        return spec.robustness(signal.T, 0, spec.parameters)


    def getErrorBand(self, controlInput, measureType, spec=None):
        
        signal = self.getSignal(controlInput)
        
        measureTypeIndex = self.requiredMeasureTypes.index(measureType)
        if spec is None:
            spec = self.fullSpec.STLFormulaObjects[measureTypeIndex]
        else:
            exec("specTemp = self."+spec+".STLFormulaObjects[measureTypeIndex]",locals(),globals())
            spec = specTemp

        # return [spec.errorBand[0](signal.T, 0), spec.errorBand[1](signal.T, 0)]
        # print(spec.errorBand)
        # print(spec.errorBand)
        # print(spec.errorBand[0](signal.T, 0))
        # print(spec.errorBand)

        # print("Here")
        # print(spec)
        # print(spec.parameters)
        # print(len(spec.parameters))
        # print(spec.robustness)

        # return spec.errorBand[0](signal.T, 0, spec.parameters) 
        return [spec.errorBand[0](signal.T, 0, spec.parameters), spec.errorBand[1](signal.T, 0, spec.parameters)] 
        
    
    def getTransferMatrixYtoU(T):
        # transfer matrix required for partial y / partial u
        I = np.eye(2)
        O = np.zeros((2,2))
        return np.block([[I if t>tau else O for tau in range(T+1)] for t in range(T+1)])


    def getRobustnessGrad(self, controlInput, measureType, spec=None):
        signal = self.getSignal(controlInput)
        
        measureTypeIndex = self.requiredMeasureTypes.index(measureType)
        if spec is None:
            spec = self.fullSpec.STLFormulaObjects[measureTypeIndex]
        else:
            exec("specTemp = self."+spec+".STLFormulaObjects[measureTypeIndex]",locals(),globals())
            spec = specTemp


        robustnessGrad = spec.robustnessGrad(signal.T, 0, spec.parameters)
        # print("RobGrad")
        # print(robustnessGrad)
        
        # print("RobGradY")
        robustnessGradY = robustnessGrad[:,0:2] 
        # print(robustnessGradY)
        
        # print("RobGradU")
        robustnessGradU = robustnessGrad[:,2:4] 
        # print(robustnessGradU)
        
        actualGrad = robustnessGradU + np.reshape( robustnessGradY.flatten()@self.transferMatrixYtoU, (len(signal.T),2))

        costFunctionGrad = 2*0.01*controlInput.T - actualGrad  

        return costFunctionGrad.T


    def costFunction(self, controlInput, measureType, spec=None):
        """
        Defines a cost function over the control sequence u such that
        the optimal u maximizes the robustness degree of the specification.
        Arguments:
            u    : a (m*T,) flattened numpy array representing a tape of control inputs
        Returns:
            J    : a scalar value indicating the degree of satisfaction of the specification.
                   (negative ==> satisfied)
        """

        # Add a small control penalty
        controlCost = 0.01 * controlInput.T @ controlInput

        # Reshape the control input to (mxT). Vector input is required for some optimization libraries
        T = int(len(controlInput)/2)
        controlInput = controlInput.reshape((2,T))

        return controlCost - self.getRobustness(controlInput, measureType, spec)





class ReachAvoidAdv(ReachAvoid):

    def __init__(self, initial_state, T=20):
        """
        Set up the example scenario with the initial state, which should be a (4,1) numpy
        array with [x,x',y,y'].
        """
        
        self.x0 = np.asarray(initial_state)
        self.T = T  # The time bound of our specification


        # vertices are given in the counterclockwise order, nx2 list, polygon should be convex
        obs1 = PolyRegion(1,'Obs.1',PolyRegion.getPolyCoordinates([3,3],6,1.5),'red',0.5)
        obs2 = PolyRegion(2,'Obs.2',PolyRegion.getPolyCoordinates([7.2,4.8],4,1),'red',0.5)
        obs3 = PolyRegion(3,'Obs.3',PolyRegion.getPolyCoordinates([4.8,7.2],4,1),'red',0.5)
        goal = PolyRegion(4,'Goal',PolyRegion.getPolyCoordinates([9,9],3,0.75,np.pi/4),'green',0.5)
        

        uMin, uMax = -1.0, 1.0 
        controlBounds = PolyRegion(5,'Control',[[uMin,uMin],[uMax,uMin],[uMax,uMax],[uMin,uMax]],'green',0.5)
                

        requiredMeasureTypes = [0,1,2,3,4] # 0: absolut, 1:Std smooth, 2:Under Approx, 3:Over Approx, 4: Reveresd
        
        # Now we'll define the STL specification. We'll do this over
        # the signal s, which is a list of x, y coordinates and the controls u_x, u_y
        # input u at each timestep. 

        # Negation can only be used with predicates, because, specs must be in "Disjunctive normal form"

        ## Avoiding the obstacle
        missObs1 = obs1.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        avoidObs1 = STLFormulas.disjunction(missObs1, requiredMeasureTypes)
        missObs2 = obs2.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        avoidObs2 = STLFormulas.disjunction(missObs2, requiredMeasureTypes)
        missObs3 = obs3.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        avoidObs3 = STLFormulas.disjunction(missObs3, requiredMeasureTypes)
        
        avoidedObstacle = [avoidObs1,avoidObs2,avoidObs3]
        self.avoidedObstacle = STLFormulas.conjunction(avoidedObstacle, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection
        

        # ## Reaching the Goal
        reachedGoal = goal.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        self.reachedGoal = STLFormulas.conjunction(reachedGoal, requiredMeasureTypes).eventually(0, self.T) # we get one STLFormula collection

        ## Bounding the control
        boundedControl = controlBounds.inControlRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        self.boundedControl = STLFormulas.conjunction(boundedControl, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection

        ## Full specification
        fullSpec = [self.avoidedObstacle, self.reachedGoal, self.boundedControl]
        self.fullSpec = STLFormulas.conjunction(fullSpec, requiredMeasureTypes)


        self.transferMatrixYtoU = ReachAvoid.getTransferMatrixYtoU(T)

        # self.regions = [obstacle, goal, aux]
        self.regions = [obs1,obs2,obs3,goal]
        self.controlBounds = controlBounds
        self.requiredMeasureTypes = requiredMeasureTypes




