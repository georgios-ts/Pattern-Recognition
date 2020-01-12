import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()


X = np.array([[4.5,1],[5,2],[5,1],[0,4.5],[0.5,4],
              [0,1],[0.5,2],[5,4],[4.5,4],[1,1]])
N = 5
classes = [i for i in range(2) for _ in range(N)]
data = pd.DataFrame({"x_1": X[:,0],
                     "x_2": X[:,1],
                     "class": classes})

plt.figure(figsize=(5,5))
sns.scatterplot(x="x_1", y="x_2", hue="class", data=data);

epsilon = .5
LIMS = (0 - epsilon, 6 + epsilon)

r = np.linspace(*LIMS, 1000)

plt.plot(r,  (-r + 3.5), linestyle='--', c='purple',alpha=0.5)
plt.plot(r,  (-r + 7.5), linestyle='--', c='purple',alpha=0.5);
w = [1, 1]
plt.quiver(1.5, 2,  w[0],  w[1], angles='xy', scale_units='xy', scale=2, alpha=.7)
plt.quiver(3.5, 4, -w[0], -w[1], angles='xy', scale_units='xy', scale=2, alpha=.7)

plt.xlim(LIMS)
plt.ylim(LIMS)
plt.gca().set_aspect('equal')

# plt.savefig("MLP.png")



def MLP(x):
    x1 = x[0]; x2 = x[1]

    h1 = np.heaviside(x2 + x1 - 3.5, 0)
    h2 = np.heaviside(-x2 - x1 + 7.5, 0)

    y = np.heaviside(h1 + h2 - 1.5, 0)
    return y > 0

for x in X:
    print("Data point ({}, {}) belongs to class 0?  {}".format(x[0], x[1], MLP(x)))
