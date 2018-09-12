"""
Learning curve sparse logistic regression
2017/2018
Author:    
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

print __doc__

import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



# dataset
beh = pd.read_excel('SCZ_mai15_full.xlsx')

X = beh.values[:, 5:35]
X_org = beh.values[:, 5:35]
X_colnames = beh.columns[5:35]
y = beh.values[:, 38]  # 'TOTAL PANSS'

ss_X = StandardScaler()
X = ss_X.fit_transform(X)
ss_y = StandardScaler()
y = ss_y.fit_transform(y.reshape(-1, 1))
y_bin = np.array(
    y >= scoreatpercentile(y, 50), dtype=np.int32)  # a classification problem !

                            
# Function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    # plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of patients", fontsize=12)
    plt.ylabel("Schizophrenia severity prediction accuracy (+/- variance)", fontsize=12)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")
    plt.legend(loc="lower right", fontsize = 12)
    plt.ylim(0.75, 1.)
    return plt


# LOG REG 
pipe = LogisticRegression(penalty='l1', verbose=False)
# which CV for the learning curve
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# fit the learning curve and plot it
title = "Lasso LogReg"
plot_learning_curve(pipe, title, X, y_bin, cv=cv, n_jobs=1, ylim=[0.7, 1])
# plt.savefig("LearnCurve_LogReg")
plt.show()





