from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel
)

def plotGP(X=None, y=None, Xstar=None, mu=None, f=None, flabel=None, sigma=None):
    NoneType = type(None)
    plt.figure(figsize=(8,6))
    if type(f) != NoneType and type(Xstar) != NoneType:
        plt.plot(Xstar, f(Xstar), 'r:', label=flabel)
    if type(X) != NoneType and type(y) != NoneType:
        plt.scatter(X, y, c='r', s=50, zorder=10, edgecolors='k', label=u'Observations')
#         plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    if type(Xstar) != NoneType and type(mu) != NoneType:
        plt.plot(Xstar, mu, 'b-', label=u'Prediction')
    if type(sigma) != NoneType:
#         plt.fill(np.concatenate([Xstar, Xstar[::-1]]),
#                  np.concatenate([mu - 1.96 * sigma,
#                                 (mu + 1.96 * sigma)[::-1]]),
#                  alpha=.2, fc='b', ec='None', label='95% confidence interval')
        plt.fill_between(Xstar.ravel(), mu - 1.96*sigma, mu + 1.96*sigma, alpha=0.2, color='b', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.show()

def prior_and_posterior(kernel):
    gp = GaussianProcessRegressor(kernel=kernel)
    plt.figure(figsize=(8, 8))
    
    # Plot prior
    plt.subplot(2, 1, 1)
    Xstar = np.linspace(0, 5, 100)
    mu, sigma = gp.predict(Xstar[:, np.newaxis], return_std=True)
    plt.plot(Xstar, mu, 'k', lw=3, zorder=9)
    plt.fill_between(Xstar, mu - sigma, mu + sigma, alpha=0.2, color='k')
    y_samples = gp.sample_y(Xstar[:, np.newaxis], 10)
    plt.plot(Xstar, y_samples, lw=1)
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

    # Generate data and fit GP
    rng = np.random.RandomState(4)
    X = rng.uniform(0, 5, 10)[:, np.newaxis]
    y = np.sin((X[:, 0] - 2.5) ** 2)
    gp.fit(X, y)

    # Plot posterior
    plt.subplot(2, 1, 2)
    mu, sigma = gp.predict(Xstar[:, np.newaxis], return_std=True)
    plt.plot(Xstar, mu, 'k', lw=3, zorder=9)
    plt.fill_between(Xstar, mu - sigma, mu + sigma, alpha=0.2, color='k')

    y_samples = gp.sample_y(Xstar[:, np.newaxis], 10)
    plt.plot(Xstar, y_samples, lw=1)
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
              % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
              fontsize=12)
    plt.show()



kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
print('kernel.hyperparameters =')
for hyperparameter in kernel.hyperparameters: print(hyperparameter)
print('\nkernel.get_params() =')
params = kernel.get_params()
for key in sorted(params): print("%s : %s" % (key, params[key]))
print('\nkernel.theta =\n', kernel.theta)  # Note: log-transformed
print('\nkernel.bounds =\n', kernel.bounds)  # Note: log-transformed


prior_and_posterior(
    kernel=1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
)

prior_and_posterior(
    kernel=1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
)

prior_and_posterior(
    kernel=1.0 * ExpSineSquared(
        length_scale=1.0,
        periodicity=3.0,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(1.0, 10.0)
    )
)

prior_and_posterior(
    kernel=ConstantKernel(0.1, (0.01, 10.0))*DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0))**2
)

prior_and_posterior(
    kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
)

def f(x):
    return x * np.sin(x)

def figure1():
    Xstar = np.atleast_2d(np.linspace(0, 10, 1000)).T

    plotGP(Xstar=Xstar, f=f, flabel=r'$f(x) = x\,\sin(x)$')

figure1()

def figure2():
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

    # Observations
    y = f(X).ravel()

    Xstar = np.atleast_2d(np.linspace(0, 10, 1000)).T
    
    gp = GaussianProcessRegressor()
    gp.fit(X, y)
    mu, sigma = gp.predict(Xstar, return_std=True)
    plotGP(X=X, y=y, Xstar=Xstar, mu=mu, f=f, flabel=r'$f(x) = x\,\sin(x)$', sigma=sigma)

figure2()

def figure3():
    X = np.atleast_2d(np.linspace(0.1, 9.9, 20)).T

    # Noisy observations
    np.random.seed(1)

    y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    
    Xstar = np.atleast_2d(np.linspace(0, 10, 1000)).T

    # Instantiate a Gaussian Process model

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    mu, sigma = gp.predict(Xstar, return_std=True)

    plotGP(X=X, y=y, Xstar=Xstar, mu=mu, f=f, flabel=r'$f(x) = x\,\sin(x)$', sigma=sigma)

figure3()

def figure4():
    X = np.atleast_2d(np.linspace(0.1, 9.9, 20)).T

    # Noisy observations
    np.random.seed(1)

    y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    
    Xstar = np.atleast_2d(np.linspace(0, 10, 1000)).T

    # Instantiate a Gaussian Process model

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-2)

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    mu, sigma = gp.predict(Xstar, return_std=True)

    plotGP(X=X, y=y, Xstar=Xstar, mu=mu, f=f, flabel=r'$f(x) = x\,\sin(x)$', sigma=sigma)

figure4()

def figure5():
    from matplotlib.colors import LogNorm

    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    # First run
    plt.figure(0)
    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X, y)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')
    plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, gp.kernel_,
                 gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()

    # Second run
    plt.figure(1)
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X, y)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')
    plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, gp.kernel_,
                 gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()

    # Plot LML landscape
    plt.figure(2)
    theta0 = np.logspace(-2, 3, 49)
    theta1 = np.logspace(-2, 0, 50)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
            for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
    LML = np.array(LML).T

    vmin, vmax = (-LML).min(), (-LML).max()
    vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
    plt.contour(Theta0, Theta1, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise-level")
    plt.title("Log-marginal-likelihood")
    plt.tight_layout()

    plt.show()

figure5()

def figure6():
    
    X = np.atleast_2d(np.linspace(0.1, 9.9, 20)).T

    # Noisy observations
    np.random.seed(1)

    y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    
    X = np.concatenate([X[:6,:], X[12:,:]], axis=0)
    y = np.concatenate([y[:6], y[12:]], axis=0)
    dy = np.concatenate([dy[:6], dy[12:]], axis=0)
    
    Xstar = np.atleast_2d(np.linspace(0, 10, 1000)).T
    
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    mu, sigma = gp.predict(Xstar, return_std=True)

    plotGP(X=X, y=y, Xstar=Xstar, mu=mu, f=f, flabel=r'$f(x) = x\,\sin(x)$', sigma=sigma)

figure6()