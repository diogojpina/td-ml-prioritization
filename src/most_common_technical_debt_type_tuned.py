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
from sklearn.metrics import accuracy_score, make_scorer
from metrics import accuracy_tuned, precision_tuned, recall_tuned, f1_tuned

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

def filterData(features, target, types):
    new_features = []
    new_target = []

    for i in range(len(features)):
        type = features[i][0]        
        if type in types:
            new_features.append(features[i])
            new_target.append(target[i])
    
    return [ new_features, new_target ]

types_set = [5153, 5496, 5113, 5595, 5328, 5535, 5353, 5213, 5510, 5092, 5430, 5532, 5347, 5590, 5579, 5098, 1861, 5093, 524, 5076, 5502, 5511, 5206, 5591, 5563, 5550, 5195, 5354, 5435, 5475, 5252, 5413, 5359, 5391, 5402, 5296, 5060, 5568, 5256, 5453, 5464, 5540, 5161, 5217, 5584, 5205, 5461, 5179, 5418, 5177, 5132, 5536, 5192, 4545, 5127, 5490, 5266, 5386, 5424, 5497, 5226, 5396, 5394, 5212, 5583, 5494, 5221, 5310, 5356]

target = data['answer'].to_numpy()

filtered_data = filterData(features, target, types_set)
filtered_features = filtered_data[0]
filtered_target = filtered_data[1]

target_xboost = [x-1 for x in filtered_target]
filtered_target = target_xboost



new_features, new_target = RandomOverSampler(random_state=0).fit_resample(filtered_features, filtered_target)
features, target = RandomOverSampler(random_state=0).fit_resample(filtered_features, filtered_target)

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

#     scorer_tuned = make_scorer(accuracy_tuned)

#     # # clf = model_selection.GridSearchCV(estimator=model, param_grid=params, scoring=['accuracy', 'f1'], refit='accuracy', n_jobs=-1, cv=kf, verbose=2)
#     # clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring=['accuracy', 'f1'], refit='accuracy', n_jobs=-1, cv=kf, verbose=2)
#     # clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring='accuracy', n_jobs=-1, cv=kf, verbose=2)
#     clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring=scorer_tuned, n_jobs=-1, cv=kf, verbose=2)
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

    labels = [1, 2, 3, 4, 5, 6]

    print('Accuracy', accuracy_tuned(target_test, target_predicted))
    print('Precision', precision_tuned(target_test, target_predicted, labels))
    print('Recall', recall_tuned(target_test, target_predicted, labels))
    print('F1', f1_tuned(target_test, target_predicted, labels))
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
        'params': {'var_smoothing': 0.657933224657568}
    },
    { 
        'name': 'KNN',
        'model': neighbors.KNeighborsClassifier(), 
        'params': {'weights': 'distance', 'p': 1, 'n_neighbors': 2, 'algorithm': 'auto'}
    },     
    { 
        'name': 'LR',
        'model': linear_model.LogisticRegression(), 
        'params': {'solver': 'newton-cg', 'penalty': 'l2', 'C': 61.11689203733466}
    },    
    { 
        'name': 'Linear',
        'model': linear_model.RidgeClassifier(),
        'params': {'solver': 'auto', 'alpha': 61.80609278803434}
    },
    { 
        'name': 'SVM',
        'model': svm.SVC(),
        'params': {'kernel': 'linear', 'C': 9.62485608807944}
    },
    { 
        'name': 'Decision Tree',
        'model': tree.DecisionTreeClassifier(),
        'params': {'splitter': 'best', 'min_weight_fraction_leaf': 0, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_leaf_nodes': 1000, 'max_features': 'sqrt', 'max_depth': 50, 'criterion': 'entropy'}
    },
    { 
        'name': 'Random Forest',
        'model': ensemble.RandomForestClassifier(),
        'params': {'n_estimators': 52, 'min_weight_fraction_leaf': 0, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 47, 'max_depth': 20, 'criterion': 'gini'}
    },
    { 
        'name': 'XGBoost',
        'model': xgb.XGBClassifier(),
        'params': {'subsample': 0.5336786183554013, 'objective': 'binary:logistic', 'n_estimators': 190, 'max_depth': 25, 'learning_rate': 0.732042607998552, 'colsample_bytree': 0.8566266174710983, 'booster': 'gbtree'}
    },
]



for model_test in models_test:
    name = model_test['name']
    model = model_test['model']
    params = model_test['params']

    print(name)
    test_method(model, params)