import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()

# load data
data = np.loadtxt('data/MLR.data')
X = data[:,:-1]
Y = data[:,-1].astype(int)

# problem parameters
N = len(X)
_, d = X.shape
n_classes = len(np.unique(Y))
K = n_classes - 1
dim = (d + 1)*K
I = np.eye(K)

# add bias (a vector of ones) to the data
F = np.ones((N, d + 1))
F[:, 1:] = X

# one hot encoding of labels - last category as a reference
T = np.zeros((N, K))
for i, y in enumerate(Y, 0):
    if y < K:
        T[i][y] = 1


# random initial weight matrix
W = np.array([0.4709,   0.1839,
             -2.5932,  -2.3215,
              3.2777,   0.0483]).reshape((d + 1, K))


# just useful to plot the data
data_t = pd.DataFrame({"x_1": X[:,0],
                       "x_2": X[:,1],
                       "classes": Y})


def draw(W, scale=1, name="weight"):
    epsilon = .5
    LIMS = (X.min() - epsilon, X.max() + epsilon)

    _, K = W.shape
    colors = sns.cubehelix_palette(n_colors=K+1)

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

draw(W, scale=1, name="initial_IWLS")

# Iterative Reweighted Least Squares
E = 1; E_prev = 0

# optimization parameters
alpha = .5
lamda = 10
epsilon = .05

while abs(E - E_prev) > epsilon:

    a = np.matmul(F, W)
    NOM = np.exp(a)
    Y_h = NOM / (1 + np.sum(NOM, axis=1))[:, None]

    # Loss
    E_prev = E
    E = -np.einsum('ij,ij->', T, np.log(Y_h)) + .5 * lamda * np.einsum('ij,ij', W, W)
    print("Loss: {}".format(E))

    # Gradient
    J = np.einsum('ij,ik->jk', F, Y_h - T) + lamda * W

    # Hessian
    H = np.einsum('ij,il,ik,km->jklm', F, F, Y_h, I) - np.einsum('ij,il,ik,im->jklm', F, F, Y_h, Y_h)
    H_2d = H.reshape((dim, dim)) + lamda * np.eye(dim)

    # Newton - Raphson update
    H_inv = np.linalg.inv(H_2d)
    W = (W.ravel() - alpha * np.dot(H_inv, J.ravel())).reshape((d + 1, K))


draw(W, scale=.3, name="final_IWLS")


def multinomial_predict(X, W):
    N, d = X.shape
    F = np.ones((N, d + 1))
    F[:, 1:] = X

    a = np.matmul(F, W)

    NOM = np.exp(a)
    Y_h = NOM / (1 + np.sum(NOM, axis=1))[:, None]

    p = np.zeros((N, n_classes))
    p[:, :-1] = Y_h
    p[:, -1] = 1 - np.sum(p, axis=1)

    return np.argmax(p,axis=1)



def multinomial_score(X, Y_true, W):
    Y_pred = multinomial_predict(X, W)

    return accuracy_score(Y_true, Y_pred)


# plot decision surfaces
M = 10000
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx = np.random.uniform(x_min, x_max, M)
yy = np.random.uniform(y_min, y_max, M)
points = np.c_[xx, yy]

Z = multinomial_predict(points, W)

data_s = pd.DataFrame({"x_1": points[:,0],
                     "x_2": points[:,1],
                     "classes": Z})


sns.scatterplot(x="x_1", y="x_2", data=data_s, \
                hue="classes", palette="Set2")

sns.scatterplot(x="x_1", y="x_2", data=data_t, \
                s=50, color='black', style="classes", legend=False);

# plt.savefig("decision_boundaries_IWLS.png")
plt.show()
