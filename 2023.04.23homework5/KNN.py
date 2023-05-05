import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

l = 10 ** 5
arr = []
x = []
y = []
dataset = {}
neighbours = 5

f = open('data.txt')
for i in range(0, 10 ** 5):
    xcoord, ycoord, n = list(map(int, f.readline().split()))
    arr.append((xcoord, ycoord))
    dataset.update({arr[i]: n})

# print(dataset.values())

f.close()

dots_train, dots_test, class_train, class_test = train_test_split(list(dataset.keys()), list(dataset.values()),
                                                                  train_size=0.7)

# print(arr_train,'\n',arr_test)

knn = KNeighborsClassifier(n_neighbors=neighbours)
knn_model = knn.fit(dots_train, class_train)
knn_predictions = knn.predict(dots_test)
# print(class_test)
# print(knn_predictions)
accuracy = accuracy_score(class_test, knn_predictions)
print(f'Accuracy: {accuracy}')

x = [[], [], [], [], [], [], [], []]
y = [[], [], [], [], [], [], [], []]
for j in range(1, 7):
    for i in range(0, len(dots_test)):
        if knn_predictions[i] == j:
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
