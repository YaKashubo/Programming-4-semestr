import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

f1 = open('data.txt')
f2 = open('classes.txt')
coords = np.array(list(map(float, f1.read().split())))
dots = np.reshape(coords, (-1, 2))
classes = np.array(list(map(int, f2.readlines())))
f1.close()
f2.close()

# for i in range(len(dots)):
#     dots[i] = np.array(cart2pol(dots[i, 0], dots[i, 1]))

dots_train, dots_test, class_train, class_test = train_test_split(dots, classes, train_size=0.25)

model = svm.SVC(kernel='linear')
svm_model = model.fit(dots_train, class_train)
svm_predictions = svm_model.predict(dots_test)
accuracy = accuracy_score(class_test, svm_predictions)
print(f'Accuracy: {accuracy}')

# for i in range(len(dots)):
#     dots[i] = np.array(pol2cart(dots[i, 0], dots[i, 1]))
# for i in range(len(dots_train)):
#     dots_train[i] = np.array(pol2cart(dots_train[i, 0], dots_train[i, 1]))
# for i in range(len(dots_test)):
#     dots_test[i] = np.array(pol2cart(dots_test[i, 0], dots_test[i, 1]))

fig1 = plt.figure(1)
fig1.suptitle('Classes')
plt.scatter(dots[:, 0], dots[:, 1], c=classes, s=0.2)

fig2 = plt.figure(2)
fig2.suptitle('Results of classification')
plt.scatter(dots_train[:, 0], dots_train[:, 1], c=class_train, s=0.2)
plt.scatter(dots_test[:, 0], dots_test[:, 1], c=svm_predictions, s=0.2)
plt.show()
