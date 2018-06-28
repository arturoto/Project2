import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD


    k = np.unique(y)
    d = X.shape[1]
    mean_vector = []
    mean_vect_total = []
    cov_vector = []

    X_y = np.hstack([X, y])

    # for the means matrix
    for j in range(len(k)):
    	for i in range(d):


    		mean_vector.append(np.mean(X_y[X_y[:, -1] == k[j]][:, i]))

    means = np.array(mean_vector).reshape(len(k), d)

    # for the covariance matrix
    for i in range(d):
    	mean_vect_total.append(np.mean(X[:, i]))

    for i in range(d):
    	for j in range(d):
    		cov_vector.append(np.dot(np.subtract(X[:, i], mean_vect_total[i]), np.subtract(X[:, j], mean_vect_total[j]).T)/ len(X[:, i]))

    covmat = np.array(cov_vector).reshape(d, d)

    #print(covmat)

    return means,covmat


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A k x d matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    k = np.unique(y)
    d = X.shape[1]
    mean_vector = []
    mean_vect_total = []
    cov_vector = []
    covmats = []
    cov_vec_vec = []

    X_y = np.hstack([X, y])

    # for the means matrix
    for j in range(len(k)):
    	for i in range(d):
    		mean_vector.append(np.mean(X_y[X_y[:, -1] == k[j]][:, i]))

    means = np.array(mean_vector).reshape(len(k), d)


    for ks in range(len(k)):

        covmats.append(np.cov(X[X_y[:, -1] == k[ks]].T))


    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    inverseSig = np.linalg.inv(covmat)

    probs = []
    ypred = []
    ys = np.unique(ytest)


    for j in range(len(ytest[:, 0])):

        probs = []

        for i in range(len(means[:, 0])):

            probs.append(np.dot(np.dot(np.subtract(Xtest[j, :], means[i, :]), inverseSig), np.subtract(Xtest[j, :], means[i, :])))

        probsNP = np.array(probs)

        try:
            if ys == np.array([0.]):
                ypred.append(0.)
            else:
                ypred.append(ys[np.argmin(probsNP)])
        except:
    
            ypred.append(ys[np.argmin(probsNP)])

    ypred = np.array(ypred).reshape(len(ytest), 1)

    acc = np.mean((ypred == ytest))

    return acc,ypred


def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    #print(covmats)
    inverseSig = []
    for i in range(len(covmats)):
        inverseSig.append(np.linalg.inv(covmats[i]))

    probs = []
    ypred = []
    ys = np.unique(ytest)



    for j in range(len(ytest[:, 0])):
        probs = []
        for i in range(len(means[:, 0])):
            #print(j)
            probs.append(np.dot(np.dot(np.subtract(Xtest[j, :], means[i, :]), inverseSig[i]),\
                                         np.subtract(Xtest[j, :], means[i, :])))


        probsNP = np.array(probs)


        try:
            if ys == np.array([0.]):
                ypred.append(0.)
            else:
                ypred.append(ys[np.argmin(probsNP)])
                
        except:
    
            ypred.append(ys[np.argmin(probsNP)])


    ypred = np.array(ypred).reshape(len(ytest), 1)

    acc = np.mean((ypred == ytest))

    return acc,ypred



def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD   

    # least square estimate from Linear Models slides Part 1 Page 8 
    w = np.dot(np.linalg.inv(np.dot(X.T,np.array(X))), np.dot(X.T,np.array(y)))


    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD

    # Ridge estimate from Linear Models slides Part 1 Page 20

    ident_d = np.identity(X.shape[1])

    w = np.dot(np.linalg.inv(np.add(np.multiply(lambd, ident_d), np.dot(X.T, X))), np.dot(X.T, y))
    

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    # Xtest is Nxd and w is dx1 so for the dot product to work, Xtest * w
    # Equation 3 from assignment 
    y_pred = np.dot(Xtest, w)

    mse = (np.sum(np.square(ytest - y_pred))) / Xtest.shape[0]

    return mse



def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    

    # Equation 4 
    y_pred = np.dot(X, w).reshape(X.shape[0], 1)

    # print(y_prepe)

    half_sum_of_squares = np.sum(np.square(np.subtract(y_pred, y))) / 2
    half_lambda_WtW = np.multiply(lambd, np.dot(w.T, w)) /2 

    error = np.add(half_sum_of_squares, half_lambda_WtW)

    # Linear Models slides Part 1 Page 21
    WtX_i = np.dot(w.T, X.T)
    WtX_i_minus_y = WtX_i - y.T
    sum_term = np.dot(WtX_i_minus_y, X)
    error_grad = (sum_term + (lambd * w)).flatten()
    #print(error_grad)



    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD

    Xp = np.array([1.0] * x.shape[0]).reshape(x.shape[0], 1)

    for i in range(p):
    	next_Column = np.array(x ** (i + 1)).reshape(x.shape[0], 1)
    	Xp = np.hstack((Xp, next_Column))


    return Xp





# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses4)]
# REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()










