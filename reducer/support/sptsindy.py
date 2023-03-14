import numpy as np

def build_rhs(y, u, coefs, mask, total_dim, poly_order, include_sine=False):

    y = np.append(y, u)
    rhs = []

    # for constant
    if mask[0]: rhs.append(coefs[0])
    
    # for cross terms
    count = 1
    for i in range(total_dim):
        if mask[count]: rhs.append(coefs[count]*y[i])
        count += 1

    if poly_order > 1:
        for i in range(total_dim):
            for j in range(i, total_dim):
                if mask[count]: rhs.append(coefs[count]*y[i]*y[j])
                count += 1

    if poly_order > 2:
        for i in range(total_dim):
            for j in range(i, total_dim):
                for k in range(j, total_dim):
                    if mask[count]: rhs.append(coefs[count]*y[i]*y[j]*y[k])
                    count += 1

    # for sine terms
    if include_sine:
        for i in range(total_dim):
            rhs.append(coefs[count]*np.sin(y[i]))
            count += 1

    return sum(rhs)

def rhs(y, t, u, coefs, mask, total_dim, poly_order=2, include_sine=False):
    dydt = [build_rhs(y, u(t), coefs[i], mask[i], total_dim, poly_order=poly_order, include_sine=include_sine)
            for i in range(len(coefs))]
    return np.array(dydt)