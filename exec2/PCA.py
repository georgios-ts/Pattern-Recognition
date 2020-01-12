import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set()

# parse data from file
X = np.loadtxt('data/PCA.data', delimiter=',', usecols=(0,1,2,3))

# standarize the data -> mean = 0 and variance = 1
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# sample covariance matrix
S = np.cov(Xn.T)

# SVD
U, D, V = np.linalg.svd(S)

# keep first two components and project
p = 2
Xpca = np.dot(Xn, U[:,:p])

# plot data along principal components
species = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']
N = 50
labels = [name for name in species for _ in range(N)]
data = pd.DataFrame({"principal_1": Xpca[:,0],
                     "principal_2": Xpca[:,1],
                     "Species": labels})

sns.scatterplot(x="principal_1", y="principal_2", hue="Species", data=data)
#plt.show()
plt.savefig("PCA_Iris.png")

# find the number of principal components that explain 95% of the variance

def ratio(A, k):
    """ parital sum / total sum """
    return np.sum(A[:k]) / np.sum(A)

perc = .95
k = 1
while ratio(D, k) < perc:
    k += 1
print("{} principal components explain 95% of the variance, perc: {:.4}".format(k, ratio(D,k)))
