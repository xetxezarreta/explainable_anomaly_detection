import numpy as np
from scipy.stats import f as f_distrib

def hotelling_tsquared(components):
    x = components.transpose()    
    mean_vec = np.mean(components, axis=0)
    cov = (components - mean_vec).T.dot((components - mean_vec)) / (components.shape[0]-1)
    w = np.linalg.solve(cov, x)
    t2 = (x * w).sum(axis=0)
    return t2

def hotelling_tsquared_v2(X, num_components=10):
    import pandas as pd
    from scipy import linalg
    from scipy.special import gammaln
    from scipy.sparse import issparse
    from scipy import linalg
    from sklearn.utils.extmath import svd_flip
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import mahalanobis
    
    [U,s,V] = linalg.svd(X);     #diagonalisation
    P = V[:,0:num_components]
    I = np.diag(s)
    I = I[0:num_components,0:num_components]
    hotelling_t2 = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        a = X[i,:].dot(P)
        b = a.dot(np.linalg.inv(I))
        c = b.dot(P.T)
        d = c.dot(X[i,:].T)
        hotelling_t2[i] = d  
        
    return hotelling_t2
    