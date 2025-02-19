#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score
import mnist_reader
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2)

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x_train = x_train/255.
x_test = x_test/255.

# stack together for next step
X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))

# one-hot encoding
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

# number of training set
m0 = 60000
m_test = X.shape[0] - m0
X_train, X_test = X[:m0].T, X[m0:].T
Y_train, Y_test = Y_new[:, :m0], Y_new[:, m0:]

# shuffle training set
shuffle_index = np.random.permutation(m0)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
X_train.shape,Y_train.shape

n_f = X_train.shape[0]
n_x = X_train.shape[1]
digits = 10

Y_hat = np.empty(shape=(digits,n_x))
lr = 1
lra = 1

#activation
def nonlin(x,ap,m):
    xdep = (-27*(x*m)/2/ap+np.sqrt(
        729*((x*m)/ap)**2+108/ap**3)/2)
    return xdep**(-1./3)/ap-xdep**(1./3)/3
def gradfn_a(x,ap):
    return -x**3/(1+3*ap*x**2)
def gradfn_m(x,y,ap):
    return x/(1+3*ap*y**2)
def gradfn_wb(x,ap,m):
    return m/(1+3*ap*x**2)

# training
permutation = np.random.permutation(X_train.shape[1])

X_train_shuffled = X_train[:, permutation]
Y_train_shuffled = Y_train[:, permutation]
batches=100
batch_size = 600
iterc = 50

figloss, axloss = plt.subplots()
figacc, axacc = plt.subplots()
markers = ['+','.']

##for a,marker in [(np.ones((digits,1)),'+'),
##                   (5*np.ones((digits,1)),'.')]:
# initialization
w = np.random.randn(digits,n_f) * np.sqrt(1. / n_f)
b = np.zeros((digits,1))
m = b#np.random.random((digits,1)) * np.sqrt(1. / digits)
wl = w
bl = b

L=np.empty(shape=(iterc,1))
Ll=np.empty(shape=(iterc,1))
a_track = np.empty(shape=(digits,iterc))
m_track = np.empty(shape=(digits,iterc))
acc = np.empty(shape=(iterc,1))
acc_l = np.empty(shape=(iterc,1))
a=1
marker='+'
for k in range(iterc):
    for j in range(batches):
        # get mini-batch
        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]#n_f,batch_size
        Y = Y_train_shuffled[:, begin:end]#digits,batch_size
        m_batch = end - begin

        xinp = np.matmul(w,X)+b#digits,batch_size
        Yb_hat = nonlin(xinp,a,m)#digits,batch_size
        Y_hat[:,begin:end] = Yb_hat 
        Y_logf = 1./(1+np.exp(-Yb_hat))
        L_sum = np.sum(np.multiply(Y, np.log(Y_logf)))
        L[k] = -(1./batch_size) * L_sum
        da = -1/m_batch*(Y_logf-Y)* gradfn_a(Yb_hat,a)
        dela = np.sum(da,axis=1,keepdims=True)
        a = a + lra*dela #Satisfy the constraint a>0
        if not np.all(a>0):
            a = a - lra*dela
            lra = lra/1.1
            print('Reducing lra to ...:',lra)
            print(lra)
            j=j-1
        else: #If a>0 is satisfied
            #tune the rest of parameters with step size set to unity
            dm = -(Y_logf-Y)* gradfn_m(xinp,nonlin(xinp,a,m),a)
            m = m + lr*np.mean(dm,axis=1,keepdims=True)
            
            dwb1 = -1/m_batch*(Y_logf-Y)* gradfn_wb(
                nonlin(xinp,a,m),a,m)
            w = w + lr*np.dot(dwb1,X.T)
            b = b + lr*np.sum(dwb1,axis=1,keepdims=True)
            # Comparison with vanilla logistic regression with step size 1
            Yl = 1./(1+np.exp(-(np.matmul(wl,X)+bl)))
            wl = wl + lr*np.dot(-(Yl-Y),X.T)
            bl = bl + lr*np.sum(-(Yl-Y),axis=1,keepdims=True)
            Ll[k] = -(1./batch_size) * np.sum(np.multiply(Y, np.log(Yl+0.05)))
            a_track[:,k]=a[:,0]
            m_track[:,k]=m[:,0]
    a_f = a
    Yt_hat = nonlin(np.dot(w,X_test)+b,a_f,m)
    Yt_logf = 1./(1+np.exp(-Yt_hat))
    Lt = -np.sum(np.multiply(Y_test, np.log(Yt_logf)),axis=1)
    print('NMI:',normalized_mutual_info_score(np.argmax(Y_test,axis=0),
                                        np.argmax(Yt_logf,axis=0)))
    #Measuring accuracy on test set and comparing with vanilla log regression
    op=np.argmax(Y_test,axis=0)-np.argmax(Yt_logf,axis=0)
    Ylt = 1./(1+np.exp(-(np.dot(wl,X_test)+bl)))
    opl = np.argmax(Y_test,axis=0)-np.argmax(Ylt,axis=0)
    acc[k] = np.count_nonzero(op)
    acc_l[k] = np.count_nonzero(opl)    
print('Continued fraction accuracy:',100-acc/100)
print('Vanilla logistic regression accuracy:',100-acc_l/100)

#Plotting
fig, ax = plt.subplots()
ax.plot(m_track[0,:],'.k',m_track[1,:],'+k',m_track[2,:],'xk',
        m_track[3,:],'1k',m_track[4,:],'2k',m_track[5,:],'3k',
        m_track[6,:],'4k',m_track[7,:],'|k',m_track[8,:],'_k');
ax.plot(m_track[9,:],'o',color=[0.5,0.5,0.5])
ax.legend(['0','1','2','3','4','5','6','7','8','9'])
plt.xlabel('iterations',fontsize=15);plt.ylabel('m',fontsize=15)
plt.grid();#plt.savefig('m_track'+str(na))

fig, ax = plt.subplots()
ax.plot(a_track[0,:],'.k',a_track[1,:],'+k',a_track[2,:],'xk',a_track[3,:],'1k',a_track[4,:],'2k',a_track[5,:],'3k',a_track[6,:],'4k',a_track[7,:],'|k',a_track[8,:],'_k');
ax.plot(a_track[9,:],'o',color=[0.5,0.5,0.5])
plt.xlabel('iterations',fontsize=15);plt.ylabel('a',fontsize=15)
ax.legend(['0','1','2','3','4','5','6','7','8','9'])
plt.grid();#plt.savefig('m_track'+str(na))

axacc.plot(100-acc/100,marker,color='k');
axloss.plot(L,marker,color='k');
print(lra,a,m)
axacc.plot(100-acc_l/100,'xk')
axacc.legend([r'$\sigma(y)$ form ($a_{{initial}}$=1)',
##              r'$\sigma(y)$ form ($a_{{initial}}$=5)',
              r'$\sigma(w^Tx)$ form'],
             fontsize=10);
axacc.set_xlabel('iterations',fontsize=15);
axacc.set_ylabel('% of correct predictions on test data',fontsize=15);
axacc.grid();

axloss.plot(Ll,'xk')
axloss.legend([r'$\sigma(y)$ form ($a_{initial}$=1)',
##              r'$\sigma(y)$ form ($a_{initial}$=5)',
               r'$\sigma(w^Tx)$ form'],
             fontsize=10);
axloss.set_xlabel('iterations',fontsize=15);
axloss.set_ylabel('Loss',fontsize=15);
axloss.grid();plt.show()
