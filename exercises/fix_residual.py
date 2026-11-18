import numpy as np
from foundation_core import Parameter, Module, initialize

def solve(direct,branch):
    return branch.copy()

if __name__ == "__main__":
    from _checks import run_local
    run_local("fix_residual", solve)
