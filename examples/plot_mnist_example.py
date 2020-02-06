"""
UMAP on the MNIST Digits dataset
--------------------------------

A simple example demonstrating how to use UMAP on a larger
dataset such as MNIST. We first pull the MNIST dataset and
then use UMAP to reduce it to only 2-dimensions for
easy visualisation.

Note that UMAP manages to both group the individual digit
classes, but also to retain the overall global structure
among the different digit classes -- keeping 1 far from
0, and grouping triplets of 3,5,8 and 4,7,9 which can
blend into one another in some cases.
"""
import sys, os
sys.path.insert(0, os.path.abspath('../'))
import umap
from umap import utils
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)

reducer = umap.UMAP(random_state=42)
data = mnist.data
# subsample
np.random.seed(42)
index = np.random.permutation(data.shape[0])[:6000]
# full sample
# index = np.arange(data.shape[0])
data = data[index]
utils.labels_global = mnist.target[index].astype(int)

# import pudb;pudb.set_trace()
embedding = reducer.fit_transform(data)


fig, ax = plt.subplots(figsize=(12, 10))
color = utils.labels_global
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
plt.colorbar()

plt.show()
