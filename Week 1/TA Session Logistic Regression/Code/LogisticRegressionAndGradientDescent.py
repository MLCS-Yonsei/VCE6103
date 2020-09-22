from copy import copy
from plotly import graph_objs as go, offline as po, figure_factory as ff, tools
import numpy as np
from sklearn.datasets import make_classification

''' Dataset '''
n_samples = 500
random_state = 120


X, Y = make_classification(
    n_samples=n_samples,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=random_state
)


''' Compute loss in parameter space (for visualization)'''
# generate grid points in parameter space
class Parameter:
    def __init__(self):
        self.gridsize = .1
        self.max = [1, 5]
        self.min = [-5, -1]
    def grid(self):
        return np.meshgrid(
            np.arange(self.min[0], self.max[0], self.gridsize),
            np.arange(self.min[1], self.max[1], self.gridsize)
        )
    def grid_and_loss(self, X, Y):
        grid = self.grid()
        loss = np.zeros_like(grid[0])
        for i in range(len(grid[0][0,:])):
            for j in range(len(grid[1][:,0])):
                loss[i, j] = np.sum(
                    np.log(
                        1+np.exp(
                            -(2*Y-1)*np.dot(
                                np.reshape([grid[0][i, j], grid[1][i, j]],[1, 2]),
                                np.transpose(X)
                            )
                        )
                    )
                )
        return grid, loss

param = Parameter()


''' Gradient descent method '''
def loss(x, y, parameters):
    return np.sum(
        np.log(
            1+np.exp(
                -(2*y-1)*np.dot(parameters.T, x.T)
            )
        )
    )

def steepest():
    max_iter = 1e4        # maximum iteration
    tol = 1e-4            # stop criteria
    learning_rate = 1.0  # \eta

    param = [np.zeros([2,1])]

    g = 1
    while np.linalg.norm(g)>tol:
        Z = 1/(1 + np.exp((2*Y-1).reshape([-1,1])*np.dot(X, param[-1])))
        g = np.dot(-np.transpose(X),Z*(2*Y-1).reshape([-1,1])) # Gradient
        p = -g/np.linalg.norm(g)
        alpha = learning_rate
        while loss(X, Y, param[-1]+alpha*p) > loss(X, Y, param[-1])+0.5*alpha*np.dot(g.T, p):
            alpha *= 0.5
        param.append(param[-1]+alpha*p)
        if len(param)>max_iter:
            break
    print('#Iterations: ', len(param)-1)
    return param

w_steepest = steepest()

def figure():
    grid, loss = param.grid_and_loss(X, Y)
    traj0 = np.reshape(w_steepest, [-1,2])
    return po.plot(
        go.Figure(
            data=[
                go.Contour(
                    x=grid[0][0,:],
                    y=grid[1][:,0],
                    z=loss,
                    opacity=0.95,
                    contours={'coloring':'heatmap'},
                    line = {'color':'rgb(255, 255, 255)'},
                    showscale=False
                ),
                go.Scatter(
                    x = traj0[:,0],
                    y = traj0[:,1],
                    mode = 'lines',
                    name = 'Steepest',
                    line = {'color':'rgb(255, 100, 100)'} # Red
                )
            ],
            layout=go.Layout(
                xaxis={'title':'w_1','range':[param.min[0], param.max[0]]},
                yaxis={'title':'w_2','range':[param.min[1], param.max[1]]},
                width=640,
                height=640
            )
        )
    )

figure()
