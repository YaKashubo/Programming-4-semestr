import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# x, y = make_classification(n_samples=10 ** 5, n_features=2,n_redundant=0,n_clusters_per_class=1,n_classes=4, class_sep=1, )
# plt.scatter(x[:, 0], x[:, 1], c=y,s=3)
# plt.show()
# f1=open("data.txt","w")
# f2=open("classes.txt","w")
#
# n=int(input())
# if n==1:
#     for i in range(len(y)):
#         f1.write(f'{x[i][0]} {x[i][1]} \n')
#         f2.write(f'{y[i]} \n')
# f1.close()
# f2.close()

neighbours = 5

f1 = open('data.txt')
f2 = open('classes.txt')
coords = np.array(list(map(float, f1.read().split())))
dots = np.reshape(coords, (-1, 2))
classes = np.array(list(map(int, f2.readlines())))
f1.close()
f2.close()

dots_train, dots_test, class_train, class_test = train_test_split(dots, classes, train_size=0.25)

fig0 = plt.figure(0)
fig0.suptitle('Train data')
plt.scatter(dots_train[:, 0], dots_train[:, 1], c=class_train, s=0.2)
plt.scatter(dots_test[:, 0], dots_test[:, 1], c='black', s=0.2)

knn = KNeighborsClassifier(n_neighbors=neighbours)
knn_model = knn.fit(dots_train, class_train)
knn_predictions = knn.predict(dots_test)
accuracy = accuracy_score(class_test, knn_predictions)
print(f'Accuracy: {accuracy}')

fig1 = plt.figure(1)
fig1.suptitle('Classes')
plt.scatter(dots[:, 0], dots[:, 1], c=classes, s=0.2)

fig2 = plt.figure(2)
fig2.suptitle('Results of classification')
plt.scatter(dots_train[:, 0], dots_train[:, 1], c=class_train, s=0.2)
plt.scatter(dots_test[:, 0], dots_test[:, 1], c=knn_predictions, s=0.2)
plt.show()
