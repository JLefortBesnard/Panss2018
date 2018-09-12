"""
Clustering -Fig 3
2017/2018
Authors: 
        Danilo Bzdok            danilobzdok (at) gmail (dot) com       
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

print __doc__


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# dataset
beh = pd.read_excel('SCZ_mai15_full.xlsx')

# To get the name of the items as x axis
i_n = pd.read_excel('name_items.xlsx')
items_array = i_n.values[:, 0:30]
items_num = []
items_name = []
items = []
for i in items_array:
    items_num.append(i[0].encode('ascii', 'ignore'))
    items_name.append(i[1].encode('ascii', 'ignore'))
    items.append([i[1].encode('ascii', 'ignore')+' '+'('+i[0].encode('ascii', 'ignore')+')'])
    
# take off coma
X_ticks1 = []
for i in items:
    x = ",".join(i)
    X_ticks1.append(x)
X_ticks = X_ticks1


X = beh.values[:, 5:35]
X_org = beh.values[:, 5:35]
X_colnames = beh.columns[5:35]
y = beh.values[:, 38]  # 'TOTAL PANSS'

ss_X = StandardScaler()
X = ss_X.fit_transform(X)



clust = KMeans(n_clusters=3, random_state=42)
X_cl = clust.fit_transform(X)
X_cl_labels = clust.labels_
my_cols = ['#CF4747', '#EA7A58', '#E4DCCB', '#524656', '#A6C4BC']

############
# plotting #
############

# centering x ticks
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


sns.set(style="white", context="talk")
# Set up the matplotlib figure
f, axarr = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
# Generate some sequential data
my_palette = ['#e74c3c'] * 7 + ['#3498db'] * 7 + ['#2ecc71'] * 16
sns.set_palette(my_palette)
X_ticks[12] = "Lack of spontaneity (N6)"

cl_mean = np.mean(X_org[X_cl_labels == 0], axis=0)
n_subs = np.sum(X_cl_labels == 0)
ax1 = sns.barplot(X_ticks, cl_mean, palette=my_palette, ax=axarr[0])
ax1.xaxis.set_ticks_position('top')
rotateTickLabels(ax1, 45, 'x')
ax1.xaxis.set_ticklabels(X_ticks, fontsize=12)
ax1.set_xlabel('%i patients' % n_subs, fontsize=16)
ax1.set_ylabel("Group 1", fontsize=16)

cl_mean = np.mean(X_org[X_cl_labels == 1], axis=0)
n_subs = np.sum(X_cl_labels == 1)
ax2 = sns.barplot(X_ticks, cl_mean, palette=my_palette, ax=axarr[1])
ax2.set_xlabel('%i patients' % n_subs, fontsize=16)
ax2.set_ylabel("Group 2", fontsize=16)

cl_mean = np.mean(X_org[X_cl_labels == 2], axis=0)
n_subs = np.sum(X_cl_labels == 2)
ax3 = sns.barplot(X_ticks, cl_mean, palette=my_palette, ax=axarr[2])
ax3.set_xlabel('%i patients' % n_subs, fontsize=16)
ax3.set_ylabel("Group 3", fontsize=16)


ax3.tick_params(axis='x',labelbottom='off')
# sns.despine(bottom=True)
plt.setp(f.axes, ylim=[0, 5.])
# plt.xticks(rotation= 90, fontsize=8)
plt.tight_layout(h_pad=3)
# plt.savefig('plots/kmeans_3cl_noYaxis.png')
plt.show()


