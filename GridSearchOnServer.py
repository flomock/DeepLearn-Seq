#!/home/go96bix/my/programs/Python-3.6.1/bin/python3.6
# import pydotplus as pydotplus
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.externals import joblib
import pandas as pd
import numpy as np
# import json
from sklearn import tree
# from IPython.display import Image
# suffix = '2017-06-23T09:42:48.759694' #Influenza
# suffix='2017-06-23T14:37:54.637627' #flavi
# suffix='2017-09-18T15:26:05.472725'  # MultiV
# suffix= 'trash/2017-09-18T11:22:45.867194' # big MultiV
# suffix = '2017-09-19T14:49:11.971644' # length 100 step 21
# suffix = '2017-09-19T15:53:46.739052' # length 100 step 21 no influ but rota
# suffix = '2017-09-19T16:58:04.893538' # length 100 only flavi
# suffix = '2017-09-19T17:07:56.940750' # length 100 smaller samples
# suffix = '2017-09-19T17:24:39.791859' # worstcase trainset small, length = 100, step 2
# suffix = '2017-09-26T16:00:47.587918'  # MUltiV random start 1000 b/chars, stepsize 5, random order
# suffix = '2017-09-28T14:37:56.166318' # MultiV for species learning stepsize 127
suffix = '2017-09-28T15:48:01.279412' # MultiV for species learning stepsize 5, 10000 samples per species

directory='/home/go96bix/Dropbox/Masterarbeit/data/machineLearning/'+suffix
Y_train = np.genfromtxt(directory + '/Y_train.csv', delimiter=',', dtype='str')
Y_test = np.genfromtxt(directory + '/Y_test.csv', delimiter=',', dtype='str')

X_train = np.genfromtxt(directory+'/X_train.csv', delimiter=',', dtype='int16')
X_test = np.genfromtxt(directory+'/X_test.csv', delimiter=',', dtype='int16')

param_grid = {
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_depth': [2, 4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'max_features': [1.0, 0.3, 0.1]
}
# est = tree.DecisionTreeClassifier(max_depth=6)
est = GradientBoostingClassifier(n_estimators=30,verbose=1)

# gs_cv = GridSearchCV(est, param_grid,n_jobs=40, verbose=10).fit(X_train, Y_train)

# joblib.dump(gs_cv, directory+'/gridsearchResult.pkl', compress = 1)
# joblib.dump(est, directory+'/GBC.pkl', compress = 1)


# print(gs_cv.best_params_)
########################
# arr = np.arange(len(X_test[0]))
# hits = np.zeros(len(X_test[0]))
# importance = np.zeros(len(X_test[0]),dtype=float)
# for i in range(0,10000):
#
#     np.random.shuffle(arr)
#     selection = arr[0:30]
#     X_train_small = X_train[:, selection]
#     X_test_small = X_test[:,selection]
#     NE = 30
#     clf = GradientBoostingClassifier(
#         n_estimators=NE,
#         learning_rate=0.1,
#         subsample=0.5,
#         random_state=42,
#         # verbose=True
#     ).fit(X_train_small, Y_train)
#     hits[selection] += 1
#     importance[selection]+=clf.feature_importances_
#
#
#     # y_true = np.array(Y_test)
#     # y_pred = clf.predict(X_test_small)
#     #
#     # # https://de.wikipedia.org/wiki/Kontingenztafel
#     # # assess
#     # table = pd.crosstab(
#     #     pd.Series(y_true),
#     #     pd.Series(y_pred),
#     #     rownames=['True'],
#     #     colnames=['Predicted'],
#     #     margins=True)
#     # print(table)
#     # print(clf.score(X_test_small, Y_test))
#
#
#     if i % 100 == 0 and i >0:
#         big_test_featureImportance = importance / hits
#         print(i)
#         print(big_test_featureImportance[big_test_featureImportance>0])
#         np.savetxt(directory+'/hits.csv', hits, fmt='%.1d', delimiter=',')
#         np.savetxt(directory+'/importance.csv', importance,  delimiter=',')
#         np.savetxt(directory+'/featureImportance_bigTest.csv',big_test_featureImportance, delimiter=',')
# big_test_featureImportance = importance / hits
# np.savetxt(directory+'/featureImportance_bigTest.csv',big_test_featureImportance, delimiter=',')
########################
est.fit(X_train,Y_train)
print(est.feature_importances_)
# hits = np.array(est.feature_importances_)
# np.savetxt(directory+'/feature_impotance_classic.csv', hits, delimiter=',')
# sub_tree_42 = est.estimators_[42, 0]

# dot_data = tree.export_graphviz(
#     est,
#     out_file=directory+'/tree_bigFile.dot', filled=True,
#     rounded=True,
#     special_characters=True,
#     proportion=True,
# )
# sub_tree_43 = est.estimators_[43, 0]
#
# dot_data = tree.export_graphviz(
#     sub_tree_43,
#     out_file=directory+'/tree2.dot', filled=True,
#     rounded=True,
#     special_characters=True,
#     proportion=True,
# )


y_true = Y_test
y_pred = est.predict(X_test)

table3=pd.crosstab(
    pd.Series(y_true),
    pd.Series(y_pred),
    rownames=['True'],
    colnames=['Predicted'],
    margins=True)
print(table3)
print(est.score(X_test, Y_test))
'''
CPU times: user 7h 33min 57s, sys: 2min 55s, total: 7h 36min 53s
Wall time: 7h 46min 5s

gs_cv.best_params_
{'learning_rate': 0.02,
 'max_depth': 4,
 'max_features': 0.1,
 'min_samples_leaf': 3}
'''

# m1 = GradientBoostingClassifier(n_estimators=1000).fit(X_train, Y_train)
# m2 = GradientBoostingClassifier(
#     n_estimators=1000,
#     learning_rate=0.1,
#     max_depth=4,
#     max_features=0.3,
#     min_samples_leaf=3).fit(X_train, Y_train)
# # clf = gs_cv.best_estimator_
# m3 = gs_cv.best_estimator_.fit(X_train, Y_train)
#
# print(m1.score(X_test, Y_test))  # 0.927
# print(m2.score(X_test, Y_test))  # 0.927
# print(m3.score(X_test, Y_test))  # 0.925
