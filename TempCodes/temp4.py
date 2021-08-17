

myLamFun = lambda x, y: x + 3*y

print(myLamFun)
print("Val=",myLamFun(3,4))

myLamFun2 = lambda x, y: myLamFun(x,y)+x+3*y

print("Val=",myLamFun(5,6))

a = 10
b = 25

myLamFun3 = lambda x, y, z=a: x + 3*y + 4*z

print(myLamFun3(2,3))
a = 34
print(myLamFun3(2,3))


def myLamFun4():
	global a
	a = a+a

# changeA = myLamFun4()
changeA = lambda : myLamFun4()

print(a)
changeA()
print(a)
changeA()
print(id(a))


# print(myLamFun4(3,4))
# print(a)



