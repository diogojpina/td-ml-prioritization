import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from scipy.stats import uniform
from sklearn import model_selection
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import gaussian_process
from sklearn import tree
from sklearn import ensemble
from sklearn import discriminant_analysis
from sklearn import neural_network
from sklearn import multiclass
from sklearn import dummy
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('csv/answers-classes.csv', low_memory=False)

features_labels = [
    'tdtype', 'ncloc', 
    'lines', 'classes', 'functions', 'statements', 'complexity', 
    'file_complexity', 'cognitive_complexity', 'comment_lines', 'comment_lines_density', 'duplicated_lines', 
    'duplicated_blocks', 'duplicated_lines_density', 'violations', 'blocker_violations', 'critical_violations', 
    'major_violations', 'minor_violations', 'bugs', 'code_smells', 'sqale_index', 
    'sqale_debt_ratio', 'sqale_rating', 'reliability_rating', 'security_rating', 'security_review_rating']
# features_labels = ['tdtype', 'lines', 'functions', 'statements', 'complexity', 'violations']
# features_labels = ['tdtype', 'lines']
features = np.nan_to_num(data[features_labels].to_numpy())


target = data['answer'].to_numpy()

target_xboost = [x-1 for x in target]
target = target_xboost


new_features, new_target = RandomOverSampler(random_state=0).fit_resample(features, target)
features, target = RandomOverSampler(random_state=0).fit_resample(features, target)

# features = MinMaxScaler().fit(features).transform(features)
# print(features)

predictores = []

# ###### DUMMY ######
# model = dummy.DummyClassifier()
# params = {
#     'strategy': ['most_frequent', 'prior', 'stratified', 'uniform', 'constant'],
#     'constant': [0, 1],
#     # 'random_state': [int(x) for x in np.linspace(0, 100, num = 10)]
# }
# predictores.append({ 'name': 'DummyClassifier', 'model': model, 'params': params })


# ##### NAIVE BAYES #####
# model = naive_bayes.GaussianNB()
# params = {
#     'var_smoothing': np.logspace(0,-9, num=100)
# }
# predictores.append({ 'name': 'GaussianNB', 'model': model, 'params': params })


# # ###### NEIGHBORS ######
# model = neighbors.KNeighborsClassifier()
# params = {
#     'n_neighbors': [int(x) for x in np.linspace(1, 20, num=20)],

#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'p': [int(x) for x in np.linspace(1, 40, num=20)]
# }
# predictores.append({ 'name': 'KNeighborsClassifier', 'model': model, 'params': params })


# ###### LOGISTIC REGRESSION ######
# model = linear_model.LogisticRegression()
# params = {
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
#     "C": [uniform.rvs(0.01, 100) for i in range(0, 10)]
# }
# predictores.append({ 'name': 'LogisticRegression', 'model': model, 'params': params })

# ###### LINEAR ######
# model = linear_model.RidgeClassifier()
# params = {
#     'alpha': [uniform.rvs(0.01, 100) for i in range(0, 10)],
#     'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
#     # 'random_state': [int(x) for x in np.linspace(0, 100, num = 6)]
# }
# predictores.append({ 'name': 'RidgeClassifier', 'model': model, 'params': params })




# ##### SVM ######
# model = svm.SVC()
# params = {
#     "C": [uniform.rvs(0.01, 10) for i in range(0, 10)],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    

#     # 'gamma': ['auto', 'scale'],
#     # 'degree': [int(x) for x in np.linspace(1, 10, num = 6)],
#     # 'coef0': [0, 1, 2, 4, 8, 16],
    
#     # 'random_state': [int(x) for x in np.linspace(0, 100, num = 6)]
# }
# predictores.append({ 'name': 'SVC', 'model': model, 'params': params })

# ###### TREES ######
# model = tree.DecisionTreeClassifier()
# params = {
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#     'min_samples_leaf': [int(x) for x in np.linspace(1, 11, num = 11)],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'splitter': ['best', 'random'],
#     'min_weight_fraction_leaf': [0, 1, 2, 4, 8],    
#     'max_leaf_nodes': [None, 20, 100, 500, 1000],
# }
# predictores.append({ 'name': 'DecisionTreeClassifier', 'model': model, 'params': params })


# ###### RANDOM FOREST ######
# model = ensemble.RandomForestClassifier()
# params = {
#     'n_estimators': [int(x) for x in np.linspace(10, 100, num = 18)],
#     'criterion': ['gini', 'entropy', 'log_loss'],    
#     'min_samples_split': [int(x) for x in np.linspace(2, 11, num = 10)],
#     'min_samples_leaf': [int(x) for x in np.linspace(1, 11, num = 11)],
#     'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
#     'max_features': [int(x) for x in np.linspace(1, 64, num = 16)],    
#     'min_weight_fraction_leaf': [0, 1, 2, 4, 8, 16]
# }
# predictores.append({ 'name': 'RandomForestClassifier', 'model': model, 'params': params })


# ###### XGBOOST ######
# model = xgb.XGBClassifier()
# params = {
#     'objective': ['binary:logistic'],
#     'booster': ['gbtree', 'gblinear', 'dart'],
#     'n_estimators': [int(x) for x in np.linspace(10, 200, num = 20)],    
#     'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
#     'learning_rate': [uniform.rvs(0.01, 1) for i in range(0, 10)],
#     'subsample': [uniform.rvs(0.01, 1) for i in range(0, 10)],
#     'colsample_bytree': [uniform.rvs(0.01, 1) for i in range(0, 10)]
    

#     # 'random_state': [42]
# }
# predictores.append({ 'name': 'XGBOOST', 'model': model, 'params': params })

# best_score = 0
# for predictor in predictores:
#     model = predictor['model']
#     params = predictor['params']

#     combinations = 1
#     for idx in params:
#         combinations = combinations * len(params[idx])

#     if (combinations <= pow(2, 10)):
#         combinations = combinations * 1.3
#     else:
#         combinations = min(combinations * 0.5, pow(2,12))


#     fold_count=5
#     kf = KFold(n_splits=fold_count, shuffle=True)

#     features_train, features_test, target_train, target_test = model_selection.train_test_split(features, target, test_size=0.2, random_state=42)

#     # # clf = model_selection.GridSearchCV(estimator=model, param_grid=params, scoring=['accuracy', 'f1'], refit='accuracy', n_jobs=-1, cv=kf, verbose=2)
#     # clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring=['accuracy', 'f1'], refit='accuracy', n_jobs=-1, cv=kf, verbose=2)
#     clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring='accuracy', n_jobs=-1, cv=kf, verbose=2)
#     search = clf.fit(features_train, target_train)

#     predictor['best_score'] = search.best_score_
#     predictor['best_params'] = search.best_params_
#     predictor['accuracy'] = max(np.nan_to_num(clf.cv_results_['mean_test_score']))

#     if search.best_score_ > best_score:
#         best_score = search.best_score_

#     with open('output/results.out', 'a') as f:
#         f.write("Predictor: " + predictor['name'] + '\n')
#         f.write("Best Score: " + str(predictor['best_score']) + '\n')
#         f.write("Best Params: " + str(predictor['best_params']) + '\n')
#         f.write("Accuracy: " + str(predictor['accuracy']) + '\n')
#         f.write('\n\n')

# for predictor in predictores:
#     print(predictor['name'])
#     print("Best Score", predictor['best_score'])
#     print("Best Params", predictor['best_params'])
#     print("Accuracy", predictor['accuracy'])

# print()
# print("Best Score")
# print(best_score)






def test_method(model, params):
    model.set_params(**params)
    target_predicted = model.fit(features_train, target_train).predict(features_test)

    print('Accuracy', accuracy_score(target_test, target_predicted))
    print('Precision', precision_score(target_test, target_predicted, average='macro'))
    print('Recall', recall_score(target_test, target_predicted, average='macro'))
    print('F1', f1_score(target_test, target_predicted, average='macro'))
    print("")


features_train, features_test, target_train, target_test = model_selection.train_test_split(features, target, test_size=0.2, random_state=42)

models_test = [
    { 
        'name': 'Dummy',
        'model': dummy.DummyClassifier(),
        'params': {'strategy': 'stratified', 'constant': 1}
    },
    { 
        'name': 'NB',
        'model': naive_bayes.GaussianNB(),
        'params': {'var_smoothing': 1e-09}
    },
    { 
        'name': 'KNN',
        'model': neighbors.KNeighborsClassifier(), 
        'params': {'weights': 'distance', 'p': 1, 'n_neighbors': 7, 'algorithm': 'auto'}
    },     
    { 
        'name': 'LR',
        'model': linear_model.LogisticRegression(), 
        'params': {'solver': 'liblinear', 'penalty': 'l2', 'C': 29.277096379469274}
    },    
    { 
        'name': 'Linear',
        'model': linear_model.RidgeClassifier(),
        'params': {'solver': 'auto', 'alpha': 76.67028446468906}
    },
    { 
        'name': 'SVM',
        'model': svm.SVC(),
        'params': {'kernel': 'linear', 'C': 1.5583798987329667}
    },
    { 
        'name': 'Decision Tree',
        'model': tree.DecisionTreeClassifier(),
        'params': {'splitter': 'random', 'min_weight_fraction_leaf': 0, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_leaf_nodes': None, 'max_features': 'auto', 'max_depth': 50, 'criterion': 'gini'}
    },
    { 
        'name': 'Random Forest',
        'model': ensemble.RandomForestClassifier(),
        'params': {'n_estimators': 94, 'min_weight_fraction_leaf': 0, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 51, 'max_depth': 40, 'criterion': 'log_loss'}
    },
    { 
        'name': 'XGBoost',
        'model': xgb.XGBClassifier(),
        'params': {'subsample': 0.9582589046992309, 'objective': 'binary:logistic', 'n_estimators': 100, 'max_depth': 25, 'learning_rate': 0.5064267401288438, 'colsample_bytree': 0.6802113366088393, 'booster': 'gbtree'}
    },
]



for model_test in models_test:
    name = model_test['name']
    model = model_test['model']
    params = model_test['params']

    print(name)
    test_method(model, params)