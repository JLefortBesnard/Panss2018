"""
I: Probing complex relationships among the PANSS items -Fig 4
2017/2018
Author: 
        Danilo Bzdok            danilobzdok (at) gmail (dot) com


II: Learning curves -SFig. 3        
2017/2018
Author:         
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

print __doc__

##############################################################
## I: Probing complex relationships among the PANSS items  ###
##############################################################

import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt
import matplotlib.patches as mpatches

#  sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# dataset
beh = pd.read_excel('SCZ_mai15_full.xlsx')

X = beh.values[:, 5:35]
X_colnames = beh.columns[5:35]
y = beh.values[:, 38]  # 'TOTAL PANSS'

ss_X = StandardScaler()
X = ss_X.fit_transform(X)
ss_y = StandardScaler()
y = ss_y.fit_transform(y.reshape(-1, 1))
y_bin = np.array(
    y >= scoreatpercentile(y, 50), dtype=np.int32)  # a classification problem !
y = np.squeeze(y_bin)

estimators_classif = [
    # linear estimators
    ['SVM-lin-l2', {'C': np.logspace(-5, +5, 11)}, LinearSVC(penalty='l2', dual=False)],
    ['LogisticRegr-l2', {'C': np.logspace(-5, +5, 11)}, LogisticRegression(penalty='l2')],
    ['RidgeClassifier-l2', {'alpha': np.logspace(-5, +5, 11)}, RidgeClassifier(solver='lsqr')],

    # non-linear estimators
    ['kNN', {'n_neighbors': np.arange(1, 26)}, KNeighborsClassifier()],
    ['AdaBoost', {'n_estimators': [50, 100, 150, 200]}, AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1))],
    ['RandForest', {'max_depth': [3, 5, 10, None], 'n_estimators': [50, 100, 200]}, RandomForestClassifier()]
]


res_cont = {}
for est_name, est_grid, est in estimators_classif:
    folder = StratifiedShuffleSplit(y=y_bin, n_iter=10, test_size=0.2)
    clf = GridSearchCV(est, est_grid)
    train_accs = []
    test_accs = []
    for train_inds, test_inds in folder:
        clf.fit(X[train_inds], y_bin[train_inds])
        y_pred = clf.score(X[train_inds], y_bin[train_inds])
        train_accs.append(y_pred)
        
        y_pred = clf.score(X[test_inds], y_bin[test_inds])
        test_accs.append(y_pred)
    
    res_cont[est_name] = {}
    res_cont[est_name]['train'] = train_accs
    res_cont[est_name]['test'] = test_accs
     
    print(est_name)
    print(np.mean(test_accs))
    
res_array = [[k, res_cont[k]['train'], res_cont[k]['test']] 
    for k in res_cont.keys()]
df = pd.DataFrame(res_array, columns=['estimator', 'train', 'test'])

# change the order of the index to get non linear next to the linear
df_ = df.copy()
df_.loc[6] = df_.loc[2]
df_.loc[7] = df_.loc[4]
df_.loc[8] = df_.loc[5]
df_.loc[9] = df_.loc[0]
df_.loc[10] = df_.loc[1]
df_.loc[11] = df_.loc[3]
df_ = df_.drop([0, 1, 2, 3, 4, 5])
df_.index = [0, 1, 2, 3, 4, 5]
# customize
df_['estimator'][0] = 'Ridge L2'
df_['estimator'][1] = 'LogReg L2'
df_['estimator'][2] = 'SVM L2'



n = len(res_cont)
plt.figure(figsize=(20, 10), dpi=80)
ax = sns.violinplot(x=None, y=None, data=df_.test, color='g')
ax = sns.violinplot(x=None, y=None, data=df_.train, color='r')
plt.axvline(x=2.5, ymin= 0, ymax = 1, linewidth=2, color='k', linestyle = '--')
plt.xlabel('Pattern-learning algorithm', fontsize=23)
plt.ylabel('schizophrenia severity prediction accuracy (+/- variance)', fontsize=23)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', labelsize=20)

# ax = sns.violinplot(x=None, y=None, data=df, palette="muted")
# plt.plot(range(n), res_array[:, 1], label='train', color='black')
# plt.plot(range(n), res_array[:, 2], label='testg', color='blue')
plt.xticks(range(n), df_.estimator.values)
test_s = mpatches.Patch(color='g', label='Test set')
train_s = mpatches.Patch(color='r', label='Train set')
plt.legend(handles=[train_s, test_s], loc='lower right', prop={'size':20})
plt.ylim(0.65, 1.07)
plt.tight_layout
# plt.legend(loc="lower right", fontsize = 14)
# plt.savefig('plots/nonlin_vs_lin_final.png', DPI=400, facecolor='white')
plt.show()




###########################
## II: Learning curves  ###
###########################


# sklearn
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
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
    plt.title(title)
    plt.ylim(0.75, 1.)
    return plt


estimators_classif = [
    # linear estimators
    ['SVM L2', {'C': np.logspace(-5, +5, 11)}, LinearSVC(penalty='l2', dual=False)],
    ['LogReg L2', {'C': np.logspace(-5, +5, 11)}, LogisticRegression(penalty='l2')],
    ['Ridge L2', {'alpha': np.logspace(-5, +5, 11)}, RidgeClassifier(solver='lsqr')],

    # non-linear estimators
    ['kNN', {'n_neighbors': np.arange(1, 10)}, KNeighborsClassifier()],
    ['AdaBoost', {'n_estimators': [50, 100, 150, 200]}, AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1))],
    ['RandForest', {'max_depth': [3, 5, 10, None], 'n_estimators': [50, 100, 200]}, RandomForestClassifier()]
]



for est_name, est_grid, est in estimators_classif:
    
    clf = GridSearchCV(est, est_grid)
    folder = StratifiedShuffleSplit(y=y_bin, n_iter=10, test_size=0.2)
    clf = GridSearchCV(est, est_grid)
    train_accs = []
    test_accs = []
    for train_inds, test_inds in folder:
        clf.fit(X[train_inds], y_bin[train_inds])
        y_pred = clf.score(X[train_inds], y_bin[train_inds])
        train_accs.append(y_pred)
        y_pred = clf.score(X[test_inds], y_bin[test_inds])
        test_accs.append(y_pred)
    best_clf = clf.best_estimator_
    
    print best_clf
    # learning curve to fit the best cv model
    pipe = best_clf
    # which CV for the learning curve, 5 splits
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # fit the learning curve and plot it
    title = "{}".format(est_name)
    plot_learning_curve(pipe, title, X, y_bin, cv=cv, n_jobs=1, ylim=[0.5, 1])
    plt.tight_layout()
    # plt.savefig("LearnCurve_{}".format(est_name))
    plt.show()



