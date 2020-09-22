from copy import copy
from plotly import graph_objs as go, offline as po, figure_factory as ff, tools
import numpy as np
from sklearn.linear_model import LinearRegression


def figure():
    X = np.reshape([1.2, 2.3, 3.1, 3.4, 4.0, 4.6, 5.5], [-1, 1])
    y = np.array([4.0, 5.6, 7.9, 8.0, 10.1, 10.4, 12.0])
    model = LinearRegression().fit(X, y)
    print('\nBeta_0 =', model.intercept_, 'Beta_1 =', model.coef_)

    data = [
        go.Scatter(
            x = [1.0, 6.0],
            y = model.coef_*[1.0, 6.0] + model.intercept_,
            mode = 'lines',
            name = 'prediction'
        ),
        go.Scatter(
            x = X.reshape([-1]),
            y = y,
            mode ='markers',
            name = 'measurement'
        )
    ]

    fig = go.Figure(data=data)

    return po.plot(fig)

figure()
