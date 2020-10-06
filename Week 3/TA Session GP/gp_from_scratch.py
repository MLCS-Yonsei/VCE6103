import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



def sqexp(x, x_, l): #squared exponential kernel
    return np.exp(-(np.linalg.norm(x-x_)**2)/(2*l**2))

def covariances(kernel, data1, data2, l):
    cov_mat = np.zeros((len(data1), len(data2)))
    for row_ in range(len(data1)):
        for col_ in range(len(data2)):
            cov_mat[row_,col_] = kernel(data1[row_], data2[col_], l)
    return cov_mat
    
X = np.array([0, 0.2, 0.3, 0.7, 1, 1.1])
Y = np.array([1, 1.5, 1.4, 0.7, 0.3, 0.1])


l = 0.2

cov_mat = covariances(sqexp, X, X, l)

x_star = np.array([1.2, 1.7]) # want to know the value at x=1.3 and 1.7
cov_mat_star = covariances(sqexp, X, x_star, l) 
cov_mat_double_star = covariances(sqexp, x_star, x_star, l)

posterior_mean = cov_mat_star.T @ np.linalg.inv(cov_mat) @ Y #assumed zero mean
posterior_cov = cov_mat_double_star - cov_mat_star.T @ np.linalg.inv(cov_mat) @ cov_mat_star

print("Results from scratch:\n",posterior_mean, "\n", posterior_cov)

kernel = RBF(length_scale=l, length_scale_bounds="fixed")
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X.reshape(-1,1), Y.reshape(-1,1))
print("Results via the library function:\n", gpr.predict(x_star.reshape(-1,1), return_cov=True))


x_r = np.linspace(0,2,100)
mu, sigma = gpr.predict(x_r[:, np.newaxis], return_std=True)
mu = mu.reshape(-1)
plt.figure()
plt.scatter(X, Y, c='k', s=50)
plt.scatter(x_star, posterior_mean, c='r', s=50)
plt.plot(x_r, mu,'k')
plt.fill_between(x_r.ravel(), mu - 1.96*sigma, mu + 1.96*sigma, alpha=0.2, color='k')
plt.show()