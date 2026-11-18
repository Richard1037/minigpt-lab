import numpy as np
from foundation_core import Parameter, Module, initialize

def solve(X):
    B,H,T,d=X.shape
    return X.reshape(B,T,H*d)

if __name__ == "__main__":
    from _checks import run_local
    run_local("fix_heads", solve)
