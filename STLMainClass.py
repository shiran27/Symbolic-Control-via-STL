##
#
# Code for specifing STL formulas and evaluating the robustness
# of STL signals, using various approximations of the min and max operators
#
##
import numpy as np
# import autograd.numpy as np
from robustnessMeasures.STLFormulaNonSmooth import STLFormulaNS
from robustnessMeasures.STLFormulaStandardApprox import STLFormulaSA
from robustnessMeasures.STLFormulaUnderApprox import STLFormulaUA
from robustnessMeasures.STLFormulaOverApprox import STLFormulaOA
from robustnessMeasures.STLFormulaReversedApprox import STLFormulaRA




class STLFormulas:

    def __init__(self, requiredMeasureTypes, robustness, errorBand, parameters, paraAddresses, paraTypes, robustnessGrad):

        self.requiredMeasureTypes = requiredMeasureTypes
        
        objectCollection = []
        for i in requiredMeasureTypes:
            if i == 0:
                objectValue = STLFormulaNS(robustness, errorBand, parameters, paraAddresses, paraTypes, robustnessGrad)
            elif i == 1:
                objectValue = STLFormulaSA(robustness, errorBand, parameters, paraAddresses, paraTypes, robustnessGrad)
            elif i == 2:
                objectValue = STLFormulaUA(robustness, errorBand, parameters, paraAddresses, paraTypes, robustnessGrad)
            elif i == 3:
                objectValue = STLFormulaOA(robustness, errorBand, parameters, paraAddresses, paraTypes, robustnessGrad)
            elif i == 4:
                objectValue = STLFormulaRA(robustness, errorBand, parameters, paraAddresses, paraTypes, robustnessGrad)

            objectCollection.append(objectValue)
        
        self.STLFormulaObjects = objectCollection

        # print("An STLFormulaSSS object was created")

   
    def constructAnObject(requiredMeasureTypes, objectCollection):
        newSTLFormulas = STLFormulas(requiredMeasureTypes,[],[],[],[],[],[])
        newSTLFormulas.STLFormulaObjects = objectCollection
        return newSTLFormulas


    def negation(self):

        objectCollection = []
        for STLFormula in self.STLFormulaObjects:
            objectCollection.append(STLFormula.negation())

        self.STLFormulaObjects = objectCollection

        return self
    
    
    def conjunction(self, secondSTLFormulasObject):

        objectCollection = []
        for i in range(len(self.STLFormulaObjects)):
            STLFormula = self.STLFormulaObjects[i]
            secondSTLFormula = secondSTLFormulasObject.STLFormulaObjects[i]
            objectCollection.append(STLFormula.conjunction(secondSTLFormula))

        self.STLFormulaObjects = objectCollection

        return self

    
    def conjunction(STLFormulasObjectList,requiredMeasureTypes):
        
        numOfTerms = len(STLFormulasObjectList)
        numOfClasses = len(STLFormulasObjectList[0].STLFormulaObjects)
        # print("Conjunction!!!: terms, classes = ",numOfTerms,numOfClasses)

        individualTerms = []
        for classNum in range(numOfClasses):
            termsForClass = []
            for termNum in range(numOfTerms):
                termsForClass.append(STLFormulasObjectList[termNum].STLFormulaObjects[classNum])
            individualTerms.append(termsForClass)

        objectCollection = []
        for classNum in range(numOfClasses):
            i = requiredMeasureTypes[classNum]
            termList = individualTerms[classNum]

            if i == 0:
                objectValue = STLFormulaNS.conjunction(termList)
            elif i == 1:
                objectValue = STLFormulaSA.conjunction(termList)
            elif i == 2:
                objectValue = STLFormulaUA.conjunction(termList)
            elif i == 3:
                objectValue = STLFormulaOA.conjunction(termList)
            elif i == 4:
                objectValue = STLFormulaRA.conjunction(termList)

            objectCollection.append(objectValue)
        
        return STLFormulas.constructAnObject(requiredMeasureTypes, objectCollection)


    def disjunction(self, secondSTLFormulasObject):

        objectCollection = []
        for i in range(len(self.STLFormulaObjects)):
            STLFormula = self.STLFormulaObjects[i]
            secondSTLFormula = secondSTLFormulasObject.STLFormulaObjects[i]
            objectCollection.append(STLFormula.disjunction(secondSTLFormula))

        self.STLFormulaObjects = objectCollection

        return self

    
    def disjunction(STLFormulasObjectList,requiredMeasureTypes):
        
        numOfTerms = len(STLFormulasObjectList)
        numOfClasses = len(STLFormulasObjectList[0].STLFormulaObjects)

        individualTerms = []
        for classNum in range(numOfClasses):
            termsForClass = []
            for termNum in range(numOfTerms):
                termsForClass.append(STLFormulasObjectList[termNum].STLFormulaObjects[classNum])
            individualTerms.append(termsForClass)


        objectCollection = []
        for classNum in range(numOfClasses):
            i = requiredMeasureTypes[classNum]
            termList = individualTerms[classNum]

            if i == 0:
                objectValue = STLFormulaNS.disjunction(termList)
            elif i == 1:
                objectValue = STLFormulaSA.disjunction(termList)
            elif i == 2:
                objectValue = STLFormulaUA.disjunction(termList)
            elif i == 3:
                objectValue = STLFormulaOA.disjunction(termList)
            elif i == 4:
                objectValue = STLFormulaRA.disjunction(termList)

            # print("in: ",i,objectValue)
            objectCollection.append(objectValue)
        
        return STLFormulas.constructAnObject(requiredMeasureTypes, objectCollection)

         
    def eventually(self, t1, t2):

        objectCollection = []
        for STLFormula in self.STLFormulaObjects:
            objectCollection.append(STLFormula.eventually(t1, t2))

        self.STLFormulaObjects = objectCollection

        return self


    def always(self, t1, t2):

        objectCollection = []
        for STLFormula in self.STLFormulaObjects:
            objectCollection.append(STLFormula.always(t1, t2))

        self.STLFormulaObjects = objectCollection

        return self

