from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, datagen, **kwargs):
        self.X, self.Y = datagen(**kwargs)
        if self.X.shape[-1]==2:
            self.x0 = self.X[:,0]
            self.x1 = self.X[:,1]

class Grid:
    def __init__(self, X, h=0.01):
        if X.shape[-1]==2:
            self.x0_min, self.x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            self.x1_min, self.x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            self.x0, self.x1 = np.meshgrid(
                np.arange(self.x0_min, self.x0_max, h),
                np.arange(self.x1_min, self.x1_max, h)
            )
            self.X = np.c_[self.x0.ravel(), self.x1.ravel()]
        elif X.shape[-1]==1:
            self.x_min, self.x_max = X[:].min() - 0.1, X[:].max() + 0.1
            self.X = np.arange(self.x_min, self.x_max, h).reshape([-1,1])

def plot_classification(model, data, hyperplane=False, **kwargs):
    
    model.fit(data.X, data.Y)
    
    grid = Grid(data.X)
    
    plt.figure(figsize=(8, 8))
    plt.clf()
    
    # get the separating hyperplane
    if hyperplane:
        w = model.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(grid.x0_min, grid.x0_max)
        yy = a * xx - (model.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors (margin away from hyperplane in direction
        # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
        # 2-d.
        margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        # plot the line, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

    prediction = model.predict(grid.X)

    # Put the result into a color plot
    plt.pcolormesh(grid.x0, grid.x1, prediction.reshape(grid.x0.shape), cmap=plt.cm.Paired)

    plt.xlim(grid.x0_min, grid.x0_max)
    plt.ylim(grid.x1_min, grid.x1_max)

    plt.xticks(())
    plt.yticks(())
    if len(kwargs) != 0:
        plt.xlabel(kwargs)
    
    try:
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
    except:
        pass
    plt.scatter(data.x0, data.x1, c=data.Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.show()


from sklearn.datasets import make_blobs

### Linearly separable data (=data0)
data0 = Data(make_blobs, centers=2, random_state=3)

# plt.figure(figsize=(8, 8))
# plt.scatter(data0.x0, data0.x1, c=data0.Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xticks(())
# plt.yticks(())
# plt.show()

from sklearn.svm import SVC

for idx in range(1,11):
    plot_classification(SVC(kernel='linear', C=1e12, max_iter=idx), data0, hyperplane=True, Iteration=idx)
    



''' Hard margin '''
plot_classification(SVC(kernel='linear', C=1e12, max_iter=1e6), data0, hyperplane=True)

''' Soft margin '''
plot_classification(SVC(kernel='linear', C=0.1), data0, hyperplane=True)

### Not linearly separable data (=data1)
data1 = Data(make_blobs, centers=2, random_state=121)

plt.figure(figsize=(8, 8))
plt.scatter(data1.x0, data1.x1, c=data1.Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.show()


''' Hard margin '''
plot_classification(SVC(kernel='linear', C=1e12, max_iter=1e6), data1, hyperplane=True)

''' Soft margin '''
plot_classification(SVC(kernel='linear', C=0.1), data1, hyperplane=True)


### Kernel tricks
data2 = Data(make_blobs, centers=4, random_state=8)
data2.Y = data2.Y % 2

plt.figure(figsize=(8, 8))
plt.scatter(data2.x0, data2.x1, c=data2.Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.show()

plot_classification(SVC(kernel='linear', C=1.0), data2, hyperplane=True)


''' Polynomial kernel with q = 2 '''
plot_classification(SVC(kernel='poly', degree=2, C=0.1), data2)
''' Polynomial kernel with q = 3 '''
plot_classification(SVC(kernel='poly', degree=3, C=0.1), data2)
''' Gaussian (radial basis function) kernel '''
plot_classification(SVC(kernel='rbf', C=1.0), data2)