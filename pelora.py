# Pelora - a supervised algorithm for grouping predictor variable
import numpy as np

"""
x: Numeric matrix of explanatory variables (p variables in columns, n cases in
rows). For example, these can be microarray gene expression data which should
be grouped

y: Numeric matrix of explanatory variables (p variables in columns, n cases in
rows). For example, these can be microarray gene expression data which should
be grouped

u: Numeric matrix of additional (clinical) explanatory variables (m variables in
columns, n cases in rows) that are used in the (penalized logistic regression)
prediction model, but neither grouped nor averaged. For example, these can be
’traditional’ clinical variables.

noc: Integer, the number of clusters that should be searched for on the data.

lambda: Real, defaults to 1/32. Rescaled penalty parameter that should be in [0, 1]

flip: Character string, describing a method how the x (gene expression) matrix should
be sign-flipped. Possible are "pm" (the default) where the sign for each variable
is determined upon its entering into the group, "cor" where the sign for each
variable is determined a priori as the sign of the empirical correlation of that
variable with the y-vector, and "none" where no sign-flipping is carried out

standardize: Logical, defaults to TRUE. Is indicating whether the predictor variables (genes)
should be standardized to zero mean and unit variance.

trace: Integer >= 0; when positive, the output of the internal loops is provided; trace
>= 2 provides output even from the internal C routines.
"""

def pelora(x, y, u=None, noc=10, lambd=1/32, flip="pm", standardize=True, trace=1):
    ## check input
    if not isinstance(x, np.ndarray):
        raise TypeError("'x' must be a numeric matrix (e.g., gene expressions)")

    # rows of `x`
    n = x.shape[0]
    if not isinstance(y, np.ndarray) or len(y) != x.shape[0] or np.any(y == 0 or y == 1):
        raise ValueError("'y' must be a numeric vector of length n = {} with only 0/1 entries".format(x.shape[0]))
    
