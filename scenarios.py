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
from numpy import linalg as LA 

from matplotlib.patches import Rectangle
from autograd import grad

from regions import PolyRegion
from STLMainClass import STLFormulas
from simulations import DataCollector

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

        self.flushParameterAddresses()

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
        ax.axis('equal')

        for poly in self.regions:
            poly.draw(ax)

    
    def flushParameterAddresses(self):

        # fullspec
        for measureTypeIndex in range(len(self.fullSpec.STLFormulaObjects)):
            if self.fullSpec.requiredMeasureTypes[measureTypeIndex] == 0:
                continue

            paraTypes = self.fullSpec.STLFormulaObjects[measureTypeIndex].paraTypes
            paraAddresses = self.fullSpec.STLFormulaObjects[measureTypeIndex].paraAddresses
            numParatypes = len(paraTypes)
            
            paraTypesNew = [paraTypes[i] for i in range(numParatypes) if paraTypes[i]!=0]
            paraAddressesNew = [paraAddresses[i] for i in range(numParatypes) if paraTypes[i]!=0]

            self.fullSpec.STLFormulaObjects[measureTypeIndex].paraTypes = paraTypesNew
            self.fullSpec.STLFormulaObjects[measureTypeIndex].paraAddresses = paraAddressesNew


    def plotControlProfile(self, u, ax, label=None):
        ax.set_xlim((-2,2)) 
        ax.set_ylim((-2,2))
        ax.axis('equal')
        self.controlBounds.draw(ax)

        ax.plot(u[0,:], u[1,:], label=label, linestyle="-", marker=".")
        # ax.grid()
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
        
    def getErrorBandWidth(self, controlInput, measureType, kValueArray):

        signal = self.getSignal(controlInput)
        measureTypeIndex = self.requiredMeasureTypes.index(measureType)
        spec = self.fullSpec.STLFormulaObjects[measureTypeIndex]
        
        if len(kValueArray)!=0:
                    
            parameters = spec.parameters
            paraAddresses = spec.paraAddresses
            paraTypes = spec.paraTypes

            newParameters = DataCollector.setValueArray(kValueArray, parameters, paraAddresses, paraTypes)
            self.fullSpec.STLFormulaObjects[measureTypeIndex].parameters = newParameters
            spec = self.fullSpec.STLFormulaObjects[measureTypeIndex]

        return -spec.errorBand[0](signal.T, 0, spec.parameters) + spec.errorBand[1](signal.T, 0, spec.parameters)


    def tuneKParameters(self, controlInput, measureType, numInterations):

        # function and gradient
        def errorBandWidthCost(kValueArray, u=controlInput, measureType=measureType):
            return self.getErrorBandWidth(u, measureType, kValueArray)+0*kValueArray.mean() #OA
            # return self.getErrorBandWidth(u, measureType, kValueArray)+5*kValueArray.mean() #UA
            # return self.getErrorBandWidth(u, measureType, kValueArray)+0.05*kValueArray.mean() #RA

        errorBandWidthCostGrad = grad(errorBandWidthCost)
        
        
        # current k value
        measureTypeIndex = self.requiredMeasureTypes.index(measureType)
        kValueArray = [] 
        for address in self.fullSpec.STLFormulaObjects[measureTypeIndex].paraAddresses:
            parameters = self.fullSpec.STLFormulaObjects[measureTypeIndex].parameters
            kVal = DataCollector.getValue(parameters,address)
            kValueArray.append(kVal)

        kValueArray = np.asarray(kValueArray)

        # GD parameters
        precision = 0.01
        kMax = 15.0
        kMin = 0.1
        stepSize = 2

        kInit = kValueArray
        
        controlInput_mat = controlInput.reshape((2,self.T+1))
        initBand = self.getErrorBand(controlInput_mat,measureType)
        initWidth = self.getErrorBandWidth(controlInput_mat, measureType, kValueArray)
        oldWidth = initWidth
        for i in range(numInterations):
            gradVal = errorBandWidthCostGrad(kValueArray,controlInput_mat,measureType)
            newkValueArray = kValueArray - stepSize*gradVal
            newkValueArray[newkValueArray>kMax] = kMax
            newkValueArray[newkValueArray<kMin] = kMin
            kValueArray = newkValueArray

            width = self.getErrorBandWidth(controlInput_mat, measureType, kValueArray)
            normGrad = LA.norm(gradVal)
            widthImprovement = oldWidth - width
            oldWidth = width
            print("i = "+str(i)+"; errot band width = "+"{:.3f}".format(width)+"; normGrad="+"{:.4f}".format(normGrad)+"; widthImpr.="+"{:.4f}".format(widthImprovement))
            # print("k="+"".join(["{:.3f}".format(kValueArray[j])+", " for j in range(len(kValueArray))]))

            if widthImprovement<precision and normGrad<precision:
                # print(gradVal)
                print("Local optimum reached at iteration %i" %i)
                break
                
        # print("Old and new k values: ")
        print("k="+"".join(["{:.3f}".format(kInit[j])+", " for j in range(len(kInit))]))
        print("k="+"".join(["{:.3f}".format(kValueArray[j])+", " for j in range(len(kValueArray))]))
        print("Old and new errro bands: ")
        print(initBand)
        print(self.getErrorBand(controlInput_mat,measureType))
        print("Total Improvement="+str(initWidth-width))


    def getTransferMatrixYtoU(T):
        # transfer matrix required for partial y / partial u
        I = np.eye(2)
        O = np.zeros((2,2))
        # return np.block([[I if t>tau else O for tau in range(T+1)] for t in range(T+1)])
        return np.block([[I if t>tau and tau>0 else O for tau in range(T+1)] for t in range(T+1)])


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
        obs1 = PolyRegion(1,'Obs.1',PolyRegion.getPolyCoordinates([4,4],6,1.5),'red',0.5)
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

        self.flushParameterAddresses()

        self.transferMatrixYtoU = ReachAvoid.getTransferMatrixYtoU(T)

        # self.regions = [obstacle, goal, aux]
        self.regions = [obs1,obs2,obs3,goal]
        self.controlBounds = controlBounds
        self.requiredMeasureTypes = requiredMeasureTypes


class FollowTrajectory(ReachAvoid):

    def __init__(self, initial_state, T=20):
        """
        Set up the example scenario with the initial state, which should be a (4,1) numpy
        array with [x,x',y,y'].
        """
        
        self.x0 = np.asarray(initial_state)
        self.T = T  # The time bound of our specification


        # vertices are given in the counterclockwise order, nx2 list, polygon should be convex
        obs1 = PolyRegion(1,'Obs.1',PolyRegion.getPolyCoordinates([5,5],7,1.5),'red',0.5)
        goal1 = PolyRegion(2,'Goal1',PolyRegion.getPolyCoordinates([5,2],3,1.5,0.1),'green',0.5)
        goal2 = PolyRegion(3,'Goal2',PolyRegion.getPolyCoordinates([8,5],3,1.5,np.pi/2+0.1),'green',0.5)
        goal3 = PolyRegion(4,'Goal3',PolyRegion.getPolyCoordinates([5,8],3,1.5,np.pi+0.1),'green',0.5)
        
        uMin, uMax = -2.0, 2.0        
        controlBounds = PolyRegion(5,'Control',[[uMin,uMin],[uMax,uMin],[uMax,uMax],[uMin,uMax]],'green',0.5)
                
        requiredMeasureTypes = [0,1,2,3,4] # 0: absolut, 1:Std smooth, 2:Under Approx, 3:Over Approx, 4: Reveresd
        
        # Now we'll define the STL specification. We'll do this over
        # the signal s, which is a list of x, y coordinates and the controls u_x, u_y
        # input u at each timestep. 

        # Negation can only be used with predicates, because, specs must be in "Disjunctive normal form"

        ## Avoiding the obstacle
        avoidedObstacle = obs1.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        self.avoidedObstacle = STLFormulas.disjunction(avoidedObstacle, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection

        t1 = self.T//3
        t2 = 2*self.T//3
        t3 = 3*self.T//4 

        # ## Reaching the Goal
        reachedGoal1 = goal1.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        reachedGoal11 = STLFormulas.conjunction(reachedGoal1, requiredMeasureTypes).eventually(0, t1) # we get one STLFormula collection

        reachedGoal2 = goal2.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        reachedGoal22 = STLFormulas.conjunction(reachedGoal2, requiredMeasureTypes).eventually(t1, t2) # we get one STLFormula collection

        reachedGoal3 = goal3.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        # reachedGoal33 = STLFormulas.conjunction(reachedGoal3, requiredMeasureTypes).eventually(t2, t3) # we get one STLFormula collection
        reachedGoal33 = STLFormulas.conjunction(reachedGoal3, requiredMeasureTypes).eventually(t2+2, self.T) # we get one STLFormula collection

        
        reachedGoal = [reachedGoal11, reachedGoal22, reachedGoal33]
        self.reachedGoal = STLFormulas.conjunction(reachedGoal, requiredMeasureTypes) # we get one STLFormula collection

        ## Bounding the control
        boundedControl = controlBounds.inControlRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        self.boundedControl = STLFormulas.conjunction(boundedControl, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection

        ## Full specification
        fullSpec = [self.avoidedObstacle, self.reachedGoal, self.boundedControl]
        # fullSpec = [self.reachedGoal, self.boundedControl]
        # fullSpec = [self.avoidedObstacle,self.boundedControl]
        # fullSpec = [self.reachedGoal]
        self.fullSpec = STLFormulas.conjunction(fullSpec, requiredMeasureTypes)

        self.flushParameterAddresses()

        self.transferMatrixYtoU = ReachAvoid.getTransferMatrixYtoU(T)

        # self.regions = [obstacle, goal, aux]
        # self.regions = [obs1,goal1,goal2,goal3,goal4]
        self.regions = [obs1,goal1,goal2,goal3]
        self.controlBounds = controlBounds
        self.requiredMeasureTypes = requiredMeasureTypes


class FollowTrajectoryAdv(ReachAvoid):

    def __init__(self, initial_state, T=20):
        """
        Set up the example scenario with the initial state, which should be a (4,1) numpy
        array with [x,x',y,y'].
        """
        
        self.x0 = np.asarray(initial_state)
        self.T = T  # The time bound of our specification


        # vertices are given in the counterclockwise order, nx2 list, polygon should be convex
        obs1 = PolyRegion(1,'Obs.1',PolyRegion.getPolyCoordinates([5,1.25],4,1),'red',0.5)
        obs2 = PolyRegion(2,'Obs.2',PolyRegion.getPolyCoordinates([5,3.75],4,1),'red',0.5)
        obs3 = PolyRegion(3,'Obs.3',PolyRegion.getPolyCoordinates([5,6.25],4,1),'red',0.5)
        obs4 = PolyRegion(4,'Obs.4',PolyRegion.getPolyCoordinates([5,8.75],4,1),'red',0.5)
        obs5 = PolyRegion(5,'Obs.5',PolyRegion.getPolyCoordinates([7.5,5],4,1.25),'red',0.5)
        obs6 = PolyRegion(6,'Obs.6',PolyRegion.getPolyCoordinates([2.5,5],4,1.25),'red',0.5)
        goal1 = PolyRegion(7,'Goal1',PolyRegion.getPolyCoordinates([7.5,2.5],3,1.2,np.pi/4),'green',0.5)
        goal2 = PolyRegion(8,'Goal2',PolyRegion.getPolyCoordinates([7.5,7.5],3,1.2,3*np.pi/4),'green',0.5)
        goal3 = PolyRegion(9,'Goal3',PolyRegion.getPolyCoordinates([2.5,7.5],3,1.2,5*np.pi/4),'green',0.5)
        
        uMin, uMax = -2.0, 2.0        
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
        missObs4 = obs4.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        avoidObs4 = STLFormulas.disjunction(missObs4, requiredMeasureTypes)
        missObs5 = obs5.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        avoidObs5 = STLFormulas.disjunction(missObs5, requiredMeasureTypes)
        missObs6 = obs6.outRegion(requiredMeasureTypes) # we get a list of STLFormula collections 
        avoidObs6 = STLFormulas.disjunction(missObs6, requiredMeasureTypes)

        avoidedObstacle = [avoidObs1,avoidObs2,avoidObs3,avoidObs4,avoidObs5,avoidObs6]
        self.avoidedObstacle = STLFormulas.conjunction(avoidedObstacle, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection
        
        t1 = self.T//3
        t2 = 2*self.T//3
        

        # ## Reaching the Goal
        reachedGoal1 = goal1.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        reachedGoal11 = STLFormulas.conjunction(reachedGoal1, requiredMeasureTypes).eventually(0, t1) # we get one STLFormula collection

        reachedGoal2 = goal2.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        reachedGoal22 = STLFormulas.conjunction(reachedGoal2, requiredMeasureTypes).eventually(t1, t2) # we get one STLFormula collection

        reachedGoal3 = goal3.inRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        # reachedGoal33 = STLFormulas.conjunction(reachedGoal3, requiredMeasureTypes).eventually(t2, t3) # we get one STLFormula collection
        reachedGoal33 = STLFormulas.conjunction(reachedGoal3, requiredMeasureTypes).eventually(t2, self.T) # we get one STLFormula collection

        # reachedGoal = [reachedGoal11, reachedGoal22, reachedGoal33, reachedGoal44]
        reachedGoal = [reachedGoal11, reachedGoal22, reachedGoal33]
        self.reachedGoal = STLFormulas.conjunction(reachedGoal, requiredMeasureTypes) # we get one STLFormula collection

        ## Bounding the control
        boundedControl = controlBounds.inControlRegion(requiredMeasureTypes) # we get a list of STLFormula collections
        self.boundedControl = STLFormulas.conjunction(boundedControl, requiredMeasureTypes).always(0, self.T) # we get one STLFormula collection

        ## Full specification
        fullSpec = [self.avoidedObstacle, self.reachedGoal, self.boundedControl]
        # fullSpec = [self.reachedGoal, self.boundedControl]
        # fullSpec = [self.avoidedObstacle,self.boundedControl]
        # fullSpec = [self.reachedGoal]
        self.fullSpec = STLFormulas.conjunction(fullSpec, requiredMeasureTypes)

        self.flushParameterAddresses()

        self.transferMatrixYtoU = ReachAvoid.getTransferMatrixYtoU(T)

        # self.regions = [obstacle, goal, aux]
        # self.regions = [obs1,goal1,goal2,goal3,goal4]
        self.regions = [obs1,obs2,obs3,obs4,obs5,obs6,goal1,goal2,goal3]
        self.controlBounds = controlBounds
        self.requiredMeasureTypes = requiredMeasureTypes
