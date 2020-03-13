import numpy as np
from scipy import linalg

def hotelling_tsquared(components):
    x = components.transpose()    
    mean_vec = np.mean(components, axis=0)
    cov = (components - mean_vec).T.dot((components - mean_vec)) / (components.shape[0]-1)
    w = np.linalg.solve(cov, x)
    t2 = (x * w).sum(axis=0)
    return t2
    
def hotelling_and_qres(X, num_components):
    [U,s,V] = linalg.svd(X)
    P = V[:,0:num_components]
    I = np.diag(s)
    I = I[0:num_components,0:num_components]

    # hotelling t2
    hotelling_t2 = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        a = X[i,:].dot(P)
        b = a.dot(np.linalg.inv(I))
        c = b.dot(P.T)
        d = c.dot(X[i,:].T)
        hotelling_t2[i] = d  
    
    # q residuals
    qres = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        PPt = P.dot(P.T)
        identity = np.identity(PPt.shape[0])
        I_PPt = identity-PPt
        a = X[i,:].dot(I_PPt)
        b = a.dot(X[i,:].T)
        qres[i] = b

    return hotelling_t2, qres
    