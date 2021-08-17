# some general functions to help simulations

class DataCollector:

	def __init__(self, length, names=[], shortNames=[]):

		if len(shortNames)==0:
			self.shortNames = names
			self.names = names
		elif len(names)==0:
			self.names = shortNames
			self.shortNames = shortNames
		else:
			self.names = names
			self.shortNames = shortNames


		self.numOfDataTypes = len(names)
		self.numOfDataPoints = length
		# print(["Data types: ",self.numOfDataTypes,"; Data Points: ",self.numOfDataPoints])

		self.dataset = [[None for i in range(length)] for j in range(len(names))]
		# j : different data types
		# i : different data points

	def updateDataset(self, i, newData):
		# print(["Data types: ",self.numOfDataTypes,"; Data Points: ",self.numOfDataPoints])
		# print(["Data Size: ", len(newData)])
		for j in range(self.numOfDataTypes):
			self.dataset[j][i] = newData[j]

	def printDataset(self, i, auxText):
		print('Iter: '+str(i)+'; '+auxText+''.join(['; '+self.shortNames[j]+':'+'{:.4f}'.format(self.dataset[j][i]) for j in range(self.numOfDataTypes)]))
	
	def truncate(self,limit):
		self.dataset = [self.dataset[j][:limit] for j in range(self.numOfDataTypes)]		




