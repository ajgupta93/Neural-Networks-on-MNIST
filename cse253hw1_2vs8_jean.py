
# coding: utf-8

# In[1]:

# importing function

import os, struct
from array import array as pyarray
from pylab import *
from numpy import *
import numpy as np

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols))
    labels = zeros((N, 1))
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    
    # image vectors
    imvs = zeros((N, rows * cols))
    for i in range(len(ind)):
        imvs[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        for j in range(len(imvs[0])):
            imvs[i][j] = imvs[i][j] * (1./255.)

    return images, labels, imvs

# Gradient Calculation
def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)

def gradient(W, X, n, lbl, target):
    T = 0
    if lbl[n] == target:
        T = 1
        
    W_T = np.transpose(W)
    
    val = -1 * np.dot(W_T, X[n])

    y_n = 1/(1 + exp(val))
    if y_n == 1:
        y_n = 0.999999
    elif y_n == 0:
        y_n = 0.000001
        
    y_n = sigmoid(-val)
    
    return (y_n - T) * X[n]

# Gradient Descent Calculation

def gradient_desc(m, nu, T, N, X, lbl, target):
    # W_0
    W = zeros((len(X[0]), 1))
    
    for j in range(m):
        S = zeros((len(X[0]), 1))
        for i in range(N):
            S = S + gradient(W, X, i, lbl, target)
        
        W = W - (nu/(1 + j/T)) * S
    return W


# In[2]:

# Load images and labels

images, labels, imvs = load_mnist('training') 

# Take from 20,000, images with 2's and 8's

L = []
L1 = []
for i in range(20000):
    if labels[i] == 2 or labels[i] == 8:
        L.append(labels[i])
        L1.append(imvs[i])

# Append '1' to image vectors

X = zeros((len(L1), len(L1[0]) + 1, 1))

for i in range(len(L1)):
    X[i] = (np.append([1], L1[i])).reshape((len(L1[0]) + 1, 1))


# In[3]:

# Calculate gradient descent

N = len(X)
m = 100
nu = 0.01#0.6#0.8#0.001
T = 3#2#1
target = 2

W = zeros((len(X[0]), 1))
alpha = gradient(W, X, 0, L, target)

#print(alpha)
w = gradient_desc(m, nu, T, N, X, L, target)

# Show the weight image

sth = w[1:785]
#print(size(sth))
a = np.reshape(sth, (28, 28))
imshow(a)
show()


# In[4]:

print(N)


# In[5]:

for i in range(len(w)):
    if w[i] > 100:
        print(i, " ", w[i])


# In[6]:

# Load images and labels

timages, tlabels, timvs = load_mnist('testing') 

# Take from 2,000, images with 2's and 3's
tL = []
tL1 = []
for i in range(2000):
    if tlabels[i] == 2 or tlabels[i] == 3:
        tL.append(tlabels[i])
        tL1.append(timvs[i])

# number of test images
num_test = len(tL)

# Append '1' to image vectors
Y = zeros((len(tL1), len(tL1[0]) + 1, 1))

for i in range(len(tL1)):
    Y[i] = (np.append([1], tL1[i])).reshape((len(tL1[0]) + 1, 1))


# In[7]:

# Accuracy percentage function

def accuracy_perc(W, N, X, lbl, target):
    W_T = np.transpose(W)
    counter = 0
    for i in range(N):
        val = np.dot(W_T, X[i])
        #if val <= -1000:
        #    val = -1000
            
        p = sigmoid(val)#1/(1 + exp(val))
        
        if p >= 0.5 and lbl[i] == target:
            #print(lbl[i])
            counter = counter + 1
        elif p < 0.5 and lbl[i] != target:
            #print(lbl[i])
            counter = counter + 1
    return 100 * counter/N


# In[8]:

# Calculate accuracy percentage on test data

acc = accuracy_perc(w, num_test, Y, tL, 2)
print(acc)


# In[9]:

def avg_err2(W, X, minv, maxv, lbl, target):
    
    W_T = np.transpose(W)
    S = 0
    counter = 0
    y_n = 0
    t_n = 0
    
    for n in range(minv, maxv):
        
        val = np.dot(W_T, X[n])
        #if val <= -1000:
        #    val = -1000
        
        y_n = sigmoid(val)#1/(1 + exp(val))
        
        counter = counter + 1
        
        if lbl[n] == target:
            t_n = 1
            if y_n == 0:
                y_n = 0.000000001
            err_n = log(y_n)

        else:
            t_n = 0
            if y_n == 1:
                y_n = 0.999999999
            err_n = log(1 - y_n)
            
        #err_n = t_n * log(y_n) + (1 - t_n) * log(1 - y_n)

        S = S + err_n
    #print(counter)
    #print(maxv-minv)
    
    #print(counter)
    return -1 * S / counter


# In[10]:

# Gradient Descent Calculation

def gradient_desc1(m, nu, T, N, X, lbl, target):
    # W_0
    W = zeros((len(X[0]), 1))
    E = zeros((m, 1))
    
    for j in range(m):
        S = zeros((len(X[0]), 1))
        for i in range(N):
            S = np.add(S, gradient(W, X, i, lbl, target))
        
        W = W - (nu/(1 + j/T)) * S
        
        E[j] = accuracy_perc(W, N, X, lbl, target)#avg_err2(W, X, 0, N, lbl, target)
        
    return W, E


# In[11]:

Tr = X[:3500]
Trl = L[:3500]

Ho = X[3500:]
Hol = L[3500:]

Te = Y
Tel = tL

print(len(Tr))
print(len(Ho))
print(len(Te))


# In[12]:

w100, err100 = gradient_desc1(500, 0.01, 2, len(Tr), Tr, Trl, target)


# In[13]:

import matplotlib.pyplot as plt

index = 500
xrange = [i for i in range(index)]

#for x in xrange:
tr = err100[:index]

plt.xlabel('m iterations')
plt.ylabel('error rate')
plt.title('Training')
plt.plot(range(index), tr, 'g-')
plt.axis([0, 100, 0, 100])
plt.show()


# In[14]:

w100, err100 = gradient_desc1(500, 0.01, 2, len(Ho), Ho, Hol, target)


# In[15]:

import matplotlib.pyplot as plt

index = 500
xrange = [i for i in range(index)]

#for x in xrange:
ho = err100[:index]

plt.xlabel('m iterations')
plt.ylabel('error rate')
plt.title('Hold-out')
plt.plot(range(index), ho, 'b-')
plt.axis([0, 100, 0, 100])
plt.show()


# In[16]:

w100, err100 = gradient_desc1(500, 0.01, 2, len(Te), Te, Tel, target)


# In[ ]:




# In[30]:

import matplotlib.pyplot as plt

index = 500
xrange = [i for i in range(index)]

#for x in xrange:
te = err100[:index]

plt.xlabel('m iterations')
plt.ylabel('error rate')
plt.title('Test')
plt.plot(range(index), te, 'r-')
plt.axis([1, 100, 0, 20])
plt.show()


# In[19]:

plt.xlabel('m iterations')
plt.ylabel('error rate')
plt.title('Training, Hold-out, and Test')
plt.plot(range(index), tr, 'g-')
plt.plot(range(index), ho, 'b-')
plt.plot(range(index), te, 'r-')
plt.axis([1, 100, 80, 100])
plt.show()


# In[20]:

# Gradient Descent Calculation

def gradient_desc2(m, nu, T, tr, trl, ho, hol, te, tel, target):#N, X, lbl, target):
    # W_0
    W = zeros((len(X[0]), 1))
    E = zeros((m, 1))
    E1 = zeros((m, 1))
    E2 = zeros((m, 1))
    
    for j in range(m):
        S = zeros((len(X[0]), 1))
        for i in range(len(tr)):
            S = np.add(S, gradient(W, tr, i, trl, target))
        
        W = W - (nu/(1 + j/T)) * S
        
        E[j] = avg_err2(W, tr, 0, len(tr), trl, target)
        E1[j] = avg_err2(W, ho, 0, len(ho), hol, target)
        E2[j] = avg_err2(W, te, 0, len(te), tel, target)
        
    return W, E, E1, E2


# In[21]:

w, tre, hoe, tee = gradient_desc2(500, 0.01, 2, Tr, Trl, Ho, Hol, Te, Tel, target)


# In[22]:

import matplotlib.pyplot as plt

index = 500
xrange = [i for i in range(index)]

#for x in xrange:
tr1 = tre[:index]
ho1 = hoe[:index]
te1 = tee[:index]

plt.xlabel('m iterations')
plt.ylabel('loss')
plt.title('Training, Hold-out, and Test')
plt.plot(range(index), tr1, 'g-')
plt.plot(range(index), ho1, 'b-')
plt.plot(range(index), te1, 'r-')
plt.axis([0, index, 0, 10])
plt.show()


# In[23]:

# Gradient Descent Calculation

def gradient_desc3(m, nu, T, tr, trl, ho, hol, te, tel, target):#N, X, lbl, target):
    # W_0
    W = zeros((len(X[0]), 1))
    E = zeros((m, 1))
    E1 = zeros((m, 1))
    E2 = zeros((m, 1))
    counter = 0
    
    for j in range(m):
        S = zeros((len(X[0]), 1))
        for i in range(len(tr)):
            S = np.add(S, gradient(W, tr, i, trl, target))
        
        W = W - (nu/(1 + j/T)) * S
        
        E[j] = avg_err2(W, tr, 0, len(tr), trl, target)
        E1[j] = avg_err2(W, ho, 0, len(ho), hol, target)
        E2[j] = avg_err2(W, te, 0, len(te), tel, target)
        
        # Early stopping
        if j >= 10 and E1[j - 1] <= E[j]:
            counter = counter + 1
        
        if counter >= 3:
            return W, E, E1, E2, j
        
    return W, E, E1, E2, j


# In[24]:

w, tre, hoe, tee, ind = gradient_desc3(500, 0.01, 2, Tr, Trl, Ho, Hol, Te, Tel, target)


# In[25]:

print(ind)


# In[26]:

import matplotlib.pyplot as plt

index = ind
xrange = [i for i in range(index)]

#for x in xrange:
tr1 = tre[:index]
ho1 = hoe[:index]
te1 = tee[:index]

plt.xlabel('m iterations')
plt.ylabel('loss')
plt.title('Training, Hold-out, and Test')
plt.plot(range(index), tr1, 'g-')
plt.plot(range(index), ho1, 'b-')
plt.plot(range(index), te1, 'r-')
plt.axis([0, index, 0, 40])
plt.show()

print(ind)


# In[27]:

# minibatch

def minibatch(m, nu, T, tr, trl, ho, hol, te, tel, target):
    # W_0
    W = zeros((len(X[0]), 1))
    E = zeros((m, 1))
    E1 = zeros((m, 1))
    E2 = zeros((m, 1))
    counter = 0
    
    for j in range(m):
        S = zeros((len(X[0]), 1))
        for i in range(len(tr)):
            S = np.add(S, gradient(W, tr, i, trl, target))
            if i % (int)(1/10 * len(tr)) == 0:
                W = W - (nu/(1 + j/T)) * S
                S = zeros((len(X[0]), 1))
        
        W = W - (nu/(1 + j/T)) * S
        
        E[j] = avg_err2(W, tr, 0, len(tr), trl, target)
        E1[j] = avg_err2(W, ho, 0, len(ho), hol, target)
        E2[j] = avg_err2(W, te, 0, len(te), tel, target)
        
        # Early stopping
        if j >= 10 and E1[j - 1] <= E[j]:
            counter = counter + 1
        
        if counter >= 3:
            return W, E, E1, E2, j
        
    return W, E, E1, E2, j  


# In[28]:

w, tre, hoe, tee, ind = minibatch(500, 0.01, 2, Tr, Trl, Ho, Hol, Te, Tel, target)


# In[29]:

import matplotlib.pyplot as plt

index = ind
xrange = [i for i in range(index)]

#for x in xrange:
tr1 = tre[:index]
ho1 = hoe[:index]
te1 = tee[:index]

plt.xlabel('m iterations')
plt.ylabel('loss')
plt.title('Training, Hold-out, and Test')
plt.plot(range(index), tr1, 'g-')
plt.plot(range(index), ho1, 'b-')
plt.plot(range(index), te1, 'r-')
plt.axis([0, index, 0, 2])
plt.show()


# In[ ]:



