import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()

# Multiclass Least-Squares classification

# load data
data = np.loadtxt('data/MLR.data')
X = data[:,:-1]
Y = data[:,-1].astype(int)

# problem parameters
N = len(X)
_, d = X.shape
n_classes = len(np.unique(Y))


# add bias (a vector of ones) to the data
F = np.ones((N, d + 1))
F[:, 1:] = X

# one-hot encoding of labels
T = np.zeros((N, n_classes))
for i, y in enumerate(Y, 0):
    T[i][y] = 1

# just useful to plot the data
data_t = pd.DataFrame({"x_1": X[:,0],
                       "x_2": X[:,1],
                       "classes": Y})

# weight matrix
W = np.matmul(np.linalg.pinv(F), T)

def draw(W, scale=1, name="weight"):
    epsilon = .5
    LIMS = (X.min() - epsilon, X.max() + epsilon)

    _, K = W.shape
    colors = sns.cubehelix_palette(n_colors=K)

    plt.figure(figsize=(6,6))
    sns.scatterplot(x="x_1", y="x_2", hue="classes", data=data_t)

    r = np.linspace(*LIMS, 100)

    for i in range(K):
        w = W[:,i]
        print(w)
        plt.plot(r,  -1 / w[2] * (w[1]*r + w[0]), linestyle='--', c=colors[i],alpha=0.8)
        x_pos = 0
        y_pos = -1 / w[2] *  w[0]
        plt.quiver(x_pos, y_pos, w[1], w[2], angles='xy', scale_units='xy', scale=scale, alpha=.7, color=colors[i])


    plt.xlim(LIMS)
    plt.ylim(LIMS)
    plt.gca().set_aspect('equal')
    # plt.savefig("{}.png".format(name))
    plt.show()

draw(W, scale=.05, name="final_LS")


def predict_LS(X, W):
    N, d = X.shape
    F = np.ones((N, d + 1))
    F[:, 1:] = X

    Y_p = np.matmul(F, W)
    return np.argmax(Y_p, axis=1)

def score_LS(X, Y, W):
    Y_p = predict_LS(X, W)

    return accuracy_score(Y, Y_p)

M = 10000
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx = np.random.uniform(x_min, x_max, M)
yy = np.random.uniform(y_min, y_max, M)
points = np.c_[xx, yy]

Z = predict_LS(points, W)

data_s = pd.DataFrame({"x_1": points[:,0],
                     "x_2": points[:,1],
                     "classes": Z})


sns.scatterplot(x="x_1", y="x_2", data=data_s, \
                hue="classes", palette="Set2")

sns.scatterplot(x="x_1", y="x_2", data=data_t, \
                s=50, color='black', style="classes", legend=False);

# plt.savefig("decision_boundaries_LS.png")
plt.show()
