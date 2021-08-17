##
#
# This file contains classes that define different types of regions in a mission space
# Each reagion can have different coordinates,
# 
# 
##


import numpy as np
# import autograd.numpy as np
from matplotlib.patches import Polygon
from STLMainClass import STLFormulas


class PolyRegion:
    
    """
    This example involves moving a robot with single integrator dynamics past an obstacle and to a goal postion with bounded control effort. 
    It also serves as a template class for more complex examples.
    """
    
    def __init__(self, index, name, coordinates,color='blue',alpha=0.5):
        """
        Set up the polygonal region with the coordinates of the vertices, which should be given as a (2,n) numpy
        array with [[x_1,y_1],[x_2,y_2],...,[x_n,y_n]]. 
        """
        self.index = index
        self.name = name
        self.coordinates = coordinates

        self.polygon = Polygon(np.array(coordinates),color = color, alpha = alpha)

        self.color = color
        self.alpha = alpha
        
        self.loadPredicates()
        
        

    def draw(self, ax):
        ax.add_patch(Polygon(np.array(self.coordinates),color = self.color, alpha = self.alpha))
    

    def getPolyCoordinates(center,numOfSides,radius,rotation=0):
        # generate serquence of coordinates to create a polygon object afterwards
        deltaTheta = 2*np.pi/numOfSides
        X = []
        theta = 0
        for i in range(numOfSides):
            X.append([center[0]+radius*np.cos(rotation+theta), center[1]+radius*np.sin(rotation+theta)])
            theta = theta+deltaTheta

        return X





    def loadPredicates(self):
        # coordinates are assumed to be in the counter clockwise order
        self.predicates = []
        self.predicateGrads = []
        for i in range(len(self.coordinates)):
            X1 = self.coordinates[i]
            if (i+1) == len(self.coordinates):
                X2 = self.coordinates[0]
            else:
                X2 = self.coordinates[i+1]
            
            angle = np.arctan2(X2[1]-X1[1],X2[0]-X1[0])

            if (X2[0]-X1[0]) != 0:
                # y = mx + c form
                if angle>-np.pi/2 and angle<np.pi/2:
                    # y-(mx+c) > 0 required to be inside
                    m = (X2[1]-X1[1])/(X2[0]-X1[0]) # m and c values, redefined to avoid closure
                    c = X2[1] - m*X2[0] 
                    muFun = lambda x, y, m=m, c=c: y-m*x-c
                    muFunGrad = lambda x, y, m=m, c=c: [-m, 1]
                else:
                    # y-(mx+c) < 0 required to be inside
                    m = (X2[1]-X1[1])/(X2[0]-X1[0]) # m and c values, redefined to avoid closure
                    c = X2[1] - m*X2[0] 
                    muFun = lambda x, y, m=m, c=c: -y+m*x+c
                    muFunGrad = lambda x, y, m=m, c=c: [m, -1]
            else:
                # x = c form
                if X2[1]>X1[1]:
                    # x < c required to be inside
                    c = X1[0] #same as X2[0], c value, to avoid closure, redefined to avoid closure
                    muFun = lambda x, y, m=m, c=c: c - x
                    muFunGrad = lambda x, y, m=m, c=c: [-1, 0]
                else:
                    # x > c required to be inside
                    # m = 0
                    c = X1[0] #same as X2[0], c value, to avoid closure, redefined to avoid closure
                    muFun = lambda x, y, m=m, c=c: x - c
                    muFunGrad = lambda x, y, m=m, c=c: [1, 0]
                    
            self.predicates.append(muFun)
            self.predicateGrads.append(muFunGrad)
                


    def isPointInside(self,xArray,yArray):
        maxViolation = -100
        minSatisfaction = 100
        max_i, min_i, max_j, min_j = -1, -1, -1, -1
        for j in range(len(xArray)):
            inPoint = True
            muArray = []

            for i in range(len(self.predicates)):
                mu_i_j = self.predicates[i](xArray[j],yArray[j])
                print("Mu, i, j = ", [mu_i_j, i, j])
                muArray.append(mu_i_j)
                if mu_i_j < 0:
                    inPoint = False
            
            if inPoint:
                maxViolation = min(muArray)
                max_i = muArray.index(maxViolation)
                max_j = j
                # print("TTMaxViol: ",[maxViolation, max_i, max_j])

            for i in range(len(self.predicates)):    
                mu_i_j = self.predicates[i](xArray[j],yArray[j])
                if mu_i_j<minSatisfaction:
                    minSatisfaction = mu_i_j
                    min_i = i
                    min_j = j
                    # print("TTMinSatis: ",[minSatisfaction, min_i, min_j])
                
                    
        return [[maxViolation, max_i, max_j],[minSatisfaction, min_i, min_j]]
        # return [maxViolation, max_i, max_j]



    def inRegion(self,requiredMeasureTypes):

        STLFormulasCollection = []
        muFunList = []
        for i in range(len(self.predicates)):

            # mu(x,y) > 0 is a predicate
            # signal: s = [x,y,u_x,u_y]_t for t=0,1,2,...,T
            
            predicateErrorMagnitude = 0.01
            parameters = predicateErrorMagnitude # default parameter theta value 
            
            # we can add a noise term here if we want based on the magnitude of theta

            # i is passed as a default variables
            predicateRobustness = lambda s, t, theta, i=i : self.predicates[i](s[t,0],s[t,1])
            predicateRobustnessGrad = lambda s, t, theta, i=i : np.asarray([[0 for j in range(len(s[0]))] if tau!=t else np.append(self.predicateGrads[i](s[t,0],s[t,1]),[0,0]).tolist() for tau in range(len(s))])

            minBand = lambda s, t, theta : -theta
            maxBand = lambda s, t, theta : theta
            predicateErrorBand = [minBand, maxBand] # errorbands in mu(x,y)
                        
            STLFormulasForASide = STLFormulas(requiredMeasureTypes, predicateRobustness, predicateErrorBand, parameters, predicateRobustnessGrad)
            STLFormulasCollection.append(STLFormulasForASide)
        
        return STLFormulasCollection
        

    def outRegion(self,requiredMeasureTypes):

        STLFormulasCollection = []
        muFunList = []
        for i in range(len(self.predicates)):

            # mu(x,y) > 0 is a predicate
            # signal: s = [x,y,u_x,u_y]_t for t=0,1,2,...,T
            
            predicateErrorMagnitude = 0.01
            parameters = predicateErrorMagnitude # default parameter theta value 
            
            # we can add a noise term here if we want based on the magnitude of theta

            # i is passed as a default variables (only dirrence compared to inRegion() is the -ve sign in following two lines)
            predicateRobustness = lambda s, t, theta, i=i : -self.predicates[i](s[t,0],s[t,1])
            predicateRobustnessGrad = lambda s, t, theta, i=i : np.asarray([[0 for j in range(len(s[0]))] if tau!=t else (-1*np.append(self.predicateGrads[i](s[t,0],s[t,1]),[0,0])).tolist() for tau in range(len(s))])

            minBand = lambda s, t, theta : -theta
            maxBand = lambda s, t, theta : theta
            predicateErrorBand = [minBand, maxBand] # errorbands in mu(x,y)
                        
            STLFormulasForASide = STLFormulas(requiredMeasureTypes, predicateRobustness, predicateErrorBand, parameters, predicateRobustnessGrad)
            STLFormulasCollection.append(STLFormulasForASide)
        
        return STLFormulasCollection

    def inControlRegion(self,requiredMeasureTypes):

        STLFormulasCollection = []
        for i in range(len(self.predicates)):
            
            # mu(x,y) > 0 is a predicate
            # signal: s = [x,y,u_x,u_y]_t for t=0,1,2,...,T
            
            predicateErrorMagnitude = 0.01
            parameters = predicateErrorMagnitude # default parameter theta value 

            # i is passed as a default variables
            predicateRobustness = lambda s, t, theta, i=i: self.predicates[i](s[t,2],s[t,3]) # we can add a noise term here if we want based on the magnitude of theta
            predicateRobustnessGrad = lambda s, t, theta, i=i: np.asarray([[0 for i in range(len(s[0]))] if tau!=t else np.append([0,0],self.predicateGrads[i](s[t,2],s[t,3])).tolist() for tau in range(len(s))])

            minBand = lambda s, t, theta : -theta
            maxBand = lambda s, t, theta : theta
            predicateErrorBand = [minBand, maxBand] # errorband in mu(x,y)
            
            STLFormulasForASide = STLFormulas(requiredMeasureTypes, predicateRobustness, predicateErrorBand, parameters, predicateRobustnessGrad)
            STLFormulasCollection.append(STLFormulasForASide)

        return STLFormulasCollection

