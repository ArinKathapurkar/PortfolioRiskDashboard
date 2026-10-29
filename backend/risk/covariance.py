import numpy as np 
from sklearn.covariance import LedoitWolf
import pandas as pd

def covariance(holdings):
    return holdings.cov()


def lw_cov(holdings):
    lw = LedoitWolf().fit(holdings)    
    return pd.DataFrame(lw.covariance_, index=holdings.columns, columns=holdings.columns)
    
    
    

    