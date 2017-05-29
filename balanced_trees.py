### Test to see if balancing-tree training data makes for more useful feature selection

import numpy as np

def data_balancer(X, y):
    balanced_X = []
    balanced_y = []
    count = 0
    for i in range(X.shape[0]):
        if y[i] == 1:
            balanced_X.append(X[i, :])
            balanced_y.append(1)
        elif (y[i] == 0 and
              count < sum(y)):
            balanced_X.append(X[i, :])
            balanced_y.append(0)
            count += 1

    return np.array(balanced_X), np.array(balanced_y)
