import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.xlabel('cluster size')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


f = open('data1.txt')
coords = np.array(list(map(int, f.read().split())))
f.close()
dots = np.reshape(coords, (-1, 2))
# print(np.shape(dots))
# dots_test = np.array_split(dots, 400)
# print(len(dots_test[0]))
hierarchy = linkage(dots, 'centroid')
clusters = fcluster(hierarchy, 4, criterion='maxclust')

plt.figure(1)
fancy_dendrogram(hierarchy, truncate_mode='lastp', p=70)

plt.figure(2)
plt.scatter(dots[:, 0], dots[:, 1], s=0.75, c=(clusters))
# plt.scatter(dots[:,0], dots[:,1], s=0.1,c='black')
plt.show()
