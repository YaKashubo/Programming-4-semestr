import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

l = 10 ** 5
arr = []
x = []
y = []
dataset = {}

f = open('data.txt')
for i in range(0, 10 ** 5):
    x1, y1, n = list(map(int, f.readline().split()))
    arr.append((x1, y1))
    dataset.update({arr[i]: n})

f.close()
# for i in range(0, 10 ** 2):
#     x1 = random.randint(-l, l)
#     y1 = random.randint(-l, l)
#     x.append(x1)
#     y.append(y1)
#     arr.append((x1, y1))
#     if(y1>=0):
#         if(x1<=(-l/3)): dataset.update({arr[i]:1})
#         if((-l/3<x1<=l/3)): dataset.update({arr[i]:2})
#         if (x1 > (l / 3)): dataset.update({arr[i]: 3})
#     else:
#         if (x1 <= (-l / 3)): dataset.update({arr[i]: 4})
#         if ((-l / 3 < x1 <= l / 3)): dataset.update({arr[i]: 5})
#         if (x1 > (l / 3)): dataset.update({arr[i]: 6})
# print(dataset.values())

dots_train, dots_test, class_train, class_test = train_test_split(list(dataset.keys()), list(dataset.values()),
                                                                  train_size=0.7)

model = svm.SVC()
svm_model = model.fit(dots_train, class_train)
svm_predictions = svm_model.predict(dots_test)
accuracy = accuracy_score(class_test, svm_predictions)
print(f'Accuracy: {accuracy}')

x = [[], [], [], [], [], [], [], []]
y = [[], [], [], [], [], [], [], []]
for j in range(1, 7):
    for i in range(0, len(dots_test)):
        if svm_predictions[i] == j:
            x[j].append(dots_test[i][0])
            y[j].append(dots_test[i][1])
plt.scatter(x[1], y[1], s=0.5, color='pink')
plt.scatter(x[2], y[2], s=0.5, color='red')
plt.scatter(x[3], y[3], s=0.5, color='orange')
plt.scatter(x[4], y[4], s=0.5, color='purple')
plt.scatter(x[5], y[5], s=0.5, color='magenta')
plt.scatter(x[6], y[6], s=0.5, color='aquamarine')

plt.plot([-10 ** 5, 10 ** 5], [0, 0], linewidth=0.25, color='black')
plt.plot([-l / 3, -l / 3], [-10 ** 5, 10 ** 5], linewidth=0.25, color='black')
plt.plot([l / 3, l / 3], [-10 ** 5, 10 ** 5], linewidth=0.25, color='black')

plt.show()
