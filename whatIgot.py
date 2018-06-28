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

    X_y = np.hstack([X, y])

    # for the means matrix
    for j in range(len(k)):
    	for i in range(d):
    		mean_vector.append(np.mean(X_y[X_y[:, -1] == k[j]][:, i]))

    means = np.array(mean_vector).reshape(len(k), d)

    ########################## This needs fixing

    for ks in range(len(k)):
    	cov_vector = []
    	for i in range(d):
	    	for j in range(d):
	    		cov_vector.append(np.dot(np.subtract(X_y[X_y[:, -1] == k[ks]][:, i], means[ks, i]), np.subtract(X_y[X_y[:, -1] == k[ks]][:, j], means[ks, j]).T)/ len(X[:, i]))

    	covmats.append(np.array(cov_vector).reshape(d, d))
    covmats = np.array(covmats)
    #############################################


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

    # print(np.random.normal(means, covmat))

    inverseSig = np.linalg.inv(covmat)

    probs = []
    ypred = []
    ys = np.unique(ytest)

    for j in range(len(ytest[:, 0])):
    	probs = []
    	for i in range(len(means[:, 0])):
    		probs.append(np.dot(np.dot(np.subtract(Xtest[j, :], means[i, :]), inverseSig),\
	    								 np.subtract(Xtest[j, :], means[i, :])))

    	probsNP = np.array(probs)
    
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

    w = np.dot(np.linalg.inv(np.dot(newX.T,np.array(X))), np.dot(newX.T,np.array(y)))


    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD

    ident_d = np.identity(X.shape[1])

    w = np.dot(np.linalg.inv(np.add(np.multiply(lambd, ident_d), \
                                                np.dot(X.T, X))), np.dot(X.T, y))
    

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    y_pred = np.dot(Xtest, w)

    mse = np.mean(np.multiply(np.subtract(ytest, y_pred), np.subtract(ytest, y_pred)))

    return mse