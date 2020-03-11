import numpy as np
from scipy.stats import f as f_distrib

def hotelling_tsquared(components):
    x = components.transpose()    
    mean_vec = np.mean(components, axis=0)
    cov = (components - mean_vec).T.dot((components - mean_vec)) / (components.shape[0]-1)
    w = np.linalg.solve(cov, x)
    t2 = (x * w).sum(axis=0)
    return t2