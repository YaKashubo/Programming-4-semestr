# approximating z(x,y)= 10 * sin(x) * sin(y) * e^-(x^2+y^2)
# max = 1.5733357151243617
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_absolute_error

dots = 10000

x = np.linspace(-4, 4, dots)
y = np.linspace(-4, 4, dots)
z = 10 * np.sin(x) * np.sin(y) * np.exp(-(x ** 2 + y ** 2))

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, train_size=0.3)
xy_train = np.array((x_train, y_train)).T
xy_test = np.array((x_test, y_test)).T
n = 10
model = tensorflow.keras.Sequential(
    [tensorflow.keras.Input(shape=(2,)),
     Dense(units=n, activation='tanh'),
     Dense(units=1, activation='linear')])
model.compile(optimizer=Adam(0.01), loss='mean_absolute_error')
foo = model.fit(xy_train, z_train, validation_data=(xy_test, z_test), epochs=50, verbose=1)

z_pred = model.predict(xy_test)
maxim_error = max_error(z_test, z_pred)
mean_error = mean_absolute_error(z_test, z_pred)
maxim = np.max(z) / 10
print(f'Средняя абсолютная ошибка = {mean_error}')
print(f'Максимальная абсолютная ошибка = {maxim_error}')
print(f'Максимум функции/10 = {maxim}')
k = 0
for i in range(len(z_pred)):
    if abs(z_test[i] - z_pred[i]) > maxim:
        k += 1
if k != 0:
    percentage = (k / len(z_pred)) * 100
else:
    percentage = 100
print(f'Процент подходящих предсказаний = {percentage}% ')

plt.plot(foo.history['loss'])
plt.plot(foo.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# xgrid, ygrid = np.meshgrid(x, y)
# z1 = 10*np.sin(xgrid)*np.sin(ygrid)*np.exp(-(xgrid**2+ ygrid**2))
# z2 = 10*np.sin(x)*np.sin(y)*np.exp(-(x**2+ y**2))
# print(z1)
# print(z2)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(xgrid, ygrid, z1)
# plt.show()
