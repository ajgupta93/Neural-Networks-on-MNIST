import cPickle
import numpy as np
import matplotlib.pyplot as plt

n_class = 10
n_feat = 784

f = open('mnist.pkl','rb')
train_set, val_set, test_set = cPickle.load(f)

train_size = 18000
val_size = 2000
test_size = 2000

bias = np.ones((train_size,1))
train_x = train_set[0][0:train_size]
train_x = np.concatenate((bias, train_x), axis=1)
train_lbl = train_set[1][0:train_size]
train_y = np.zeros((train_size,n_class))
for i in range(0,train_size):
	train_y[i][train_lbl[i]] = 1

print train_y[1],train_lbl[1]
print train_y[2],train_lbl[2]

bias = np.ones((val_size,1))
val_x = train_set[0][train_size:train_size+val_size]
val_x = np.concatenate((bias, val_x), axis=1)
val_lbl = train_set[1][train_size:train_size+val_size]
val_y = np.zeros((val_size,n_class))
for i in range(0,val_size):
	val_y[i][val_lbl[i]] = 1

bias = np.ones((test_size,1))
test_x = test_set[0][0:test_size]
test_x = np.concatenate((bias, test_x), axis=1)
test_lbl = test_set[1][0:test_size]

n_iters = 1000
eta0 = 0.1
T = 2
w = np.ones((n_class,n_feat+1))
valsum = []
trainsum = []
testsum = []
lamda = 1

vv = []
tt = []
ts = []

for i in range(0,n_iters):
	#training
	z = np.dot(w,train_x.T)
	zmax = np.amax(z, axis=0)
	ez = np.exp(z-zmax)
	a = ez / np.sum(ez, axis=0)
	grad = np.dot(a-train_y.T,train_x)+lamda/train_size
	eta = eta0/(1+i/T)
	w = w-eta*grad
	
	#validation error
	zz = np.dot(w,val_x.T)
	zzmax = np.amax(zz, axis=0)
        ezz = np.exp(zz - zzmax)
        p = ezz / np.sum(ezz, axis=0)
        vv.append(float(np.sum(np.argmax(p, axis=0)==val_lbl))/float(val_size))


	#training error
	zz = np.dot(w,train_x.T)
	zzmax = np.amax(zz, axis=0)
        ezz = np.exp(zz - zzmax)
        p2 = ezz / np.sum(ezz, axis=0)
        tt.append(float(np.sum(np.argmax(p2, axis=0)==train_lbl))/float(train_size))

	#testing error
	zz = np.dot(w,test_x.T)
	zzmax = np.amax(zz, axis=0)
        ezz = np.exp(zz - zzmax)
        p3 = ezz / np.sum(ezz, axis=0)
        ts.append(float(np.sum(np.argmax(p3, axis=0)==test_lbl))/float(test_size))

	#loss
	sum = 0
	for i in range(0,val_size):
		if p[val_lbl[i]][i]>0:
			sum+=-np.log(p[val_lbl[i]][i])
	sum+=np.sum(w[2:])*lamda/val_size
	#print sum/val_size
	valsum.append(sum/val_size)

	sum = 0
	for i in range(0,train_size):
		if p2[train_lbl[i]][i]>0:
			sum+=-np.log(p2[train_lbl[i]][i])
	sum+=np.sum(w[2:])*lamda/train_size
	#print sum/train_size
	trainsum.append(sum/train_size)

	sum = 0
	for i in range(0,test_size):
		if p3[test_lbl[i]][i]>0:
			sum+=-np.log(p3[test_lbl[i]][i])
	sum+=np.sum(w[2:])*lamda/test_size
	#print sum/test_size
	testsum.append(sum/test_size)
	
#print allsums,[i for i in range(0,n_iters)]
plt.plot([i for i in range(0,n_iters)],trainsum)
plt.plot([i for i in range(0,n_iters)],valsum)
plt.plot([i for i in range(0,n_iters)],testsum)
plt.axis([0, n_iters, 0, 10])
plt.show()

plt.plot([i for i in range(0,n_iters)],tt)
plt.plot([i for i in range(0,n_iters)],vv)
plt.plot([i for i in range(0,n_iters)],ts)
plt.axis([0, n_iters, 0, 1])
plt.show()

#testing
zz = np.dot(w,test_x.T)
zzmax = np.amax(zz, axis=0)
ezz = np.exp(zz - zzmax)
p = ezz / np.sum(ezz, axis=0)
print float(np.sum(np.argmax(p, axis=0)==test_lbl))/float(len(test_lbl))
