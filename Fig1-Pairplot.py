"""
Pairplot -Fig 1
2017/2018
Author:    
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

print __doc__


import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pylab as plt
import seaborn as sns
from scipy.stats.stats import pearsonr

# datset
beh = pd.read_excel('SCZ_mai15_full.xlsx')

X = beh.values[:, 5:35]             # ALL data
Pos = beh.values[:, 5:12]           # POSITIVE symptoms item scores
Neg = beh.values[:, 12:19]          # NEGATIVE symptoms item scores
Gen = beh.values[:, 19:35]          # GENERAL symptoms item scores


#### ITEMS NAME ALL ###
items = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'G1', 'G2', 'G3',\
        'G4', 'G5', 'G6','G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', "G16"]
# POS
P_items = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
# NEG
N_items = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
# GEN
G_items = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10',
           'G11', 'G12', 'G13', 'G14', 'G15', "G16"]

################################################################################
## Averaging (np.mean) per positive, negative and general symptom item scores ##
################################################################################

# STANDARDIZE THE DATA
Xpng = np.array(
        [np.mean(Pos, axis=1),
        np.mean(Neg, axis=1),
        np.mean(Gen, axis=1)]
        )
ss_X = StandardScaler()
Xpng2 = ss_X.fit_transform(Xpng.T)

# plot it
items = ['Positive', 'Negative', 'General']
df = pd.DataFrame(data=Xpng2, columns=items)
# sns.pairplot(df, diag_kind="kde", kind="Reg")
sns.pairplot(df, diag_kind="kde", kind="reg")
# sns.pairplot(df)
# plt.savefig("paiplot_PNG_reg")
plt.show()


# pearsonr returns Pearson corr + 2 tailed pvalues
print "pos-neg", pearsonr(df["Positive"], df["Negative"])
print "pos-gen", pearsonr(df["Positive"], df["General"])
print "gen-neg", pearsonr(df["General"], df["Negative"])



# sns pair grid
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
# plt.savefig("PairGrid_Stand_Kde")
plt.show()       

