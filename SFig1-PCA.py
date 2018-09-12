"""
PCA
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
import copy
import joblib

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit


# dataset
beh = pd.read_excel('SCZ_mai15_full.xlsx')


X = beh.values[:, 5:35]
X_org = beh.values[:, 5:35]
X_colnames = beh.columns[5:35]


ss_X = StandardScaler()
X = ss_X.fit_transform(X)
plt.close('all')


# make item names pretty
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
X_ticks[12] = "Lack of spontaneity (N6)"

# center the x ticks
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

# PCA 5 components
decomposer = PCA(n_components=5, random_state=42)
X_comp = decomposer.fit_transform(X)

f, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(decomposer.components_, cmap='RdBu_r', square=True, cbar_kws={"shrink": .25})
ax.xaxis.set_ticks_position('top')
rotateTickLabels(ax, 45, 'x')
ax.xaxis.set_ticklabels(X_ticks, fontsize=12)
y_ticks = ['%.2f%%' % ev for ev in np.round(decomposer.explained_variance_ratio_*100, decimals=2)]
ax.yaxis.set_ticklabels(y_ticks, fontsize=12, rotation=0)
plt.ylabel('Components', fontsize=12)
plt.tight_layout()
# plt.savefig('PCA_final.png')
plt.show()

