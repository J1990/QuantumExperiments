import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas
import numpy as np
import csv

table = pandas.read_csv('DebugOutput\\result.csv')

X = table['PC1']
Y = table['PC2']
Z = table['ActImg']
classes = ['Correct Predictions', 'Incorrect Predictions']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(X, Y, Z, s = 40, c=table['ClassicalAreEqual'])
ax.set_zticks([3, 6])
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("MNIST Digit")

#legend1 = ax.legend(*scatter.legend_elements(),
#                    loc="upper right", title="Predictions")
#ax.add_artist(legend1)

plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.show()

