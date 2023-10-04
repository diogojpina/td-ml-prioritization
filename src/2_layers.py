import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import dummy
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from metrics import accuracy_tuned, f1_tuned, precision_tuned, recall_tuned
import collections.abc
import copy

class TwoLayersClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        if 'classifier' in kwargs:
            self.classifier = kwargs['classifier']
        else:
            self.classifier = RandomForestClassifier
        
        kwargs_copy = copy.deepcopy(kwargs)
        del kwargs_copy['classifier']

        self.params = copy.deepcopy(kwargs)

        self.method_layer1 = RandomForestClassifier(n_estimators=52, min_weight_fraction_leaf=0, min_samples_split=3, min_samples_leaf=1, max_features=1, max_depth=40, criterion='gini')
        self.method_layer2 = self.classifier(**kwargs_copy)        

    def fit(self, X, y=None, **fit_params):
        y_bool = [0] * len(y)
        for i in range(len(y_bool)):
            if (y[i] < 6): y_bool[i] = 1
            else: y_bool[i] = 0

        self.model_layer1 = self.method_layer1.fit(X, y_bool)
        
        X_pay = []
        y_pay = []
        for i in range(len(y)):
            if y[i] == 6: 
                continue

            y_pay.append(y[i])
            X_pay.append(X[i])


        self.model_layer2 = self.method_layer2.fit(X_pay, y_pay)

        return self
    
    def predict(self, X, y=None):
        target_predicted_layer1 = self.model_layer1.predict(X)
        target_predicted_layer2 = self.model_layer2.predict(X)

        hits = 0
        total = 0
        target_predicted = [0] * len(X)
        for i in range(len(X)):
            if (target_predicted_layer1[i] == 0):
                target_predicted[i] = 6
                continue
            target_predicted[i] = target_predicted_layer2[i]            
        
        return target_predicted

    def get_params(self, deep=True):
        params = self.method_layer2.get_params()
        params["classifier"] = self.classifier

        return params

    def set_params(self, **params):
        params_copy = copy.deepcopy(params)
        del params_copy['classifier']
        return self.method_layer2.set_params(**params_copy)
    
    def predict_proba(self, X, y=None):
        pass

data = pd.read_csv('csv/answers-classes.csv', low_memory=False)

features_labels = [
    'tdtype', 'ncloc', 
    'lines', 'classes', 'functions', 'statements', 'complexity', 
    'file_complexity', 'cognitive_complexity', 'comment_lines', 'comment_lines_density', 'duplicated_lines', 
    'duplicated_blocks', 'duplicated_lines_density', 'violations', 'blocker_violations', 'critical_violations', 
    'major_violations', 'minor_violations', 'bugs', 'code_smells', 'sqale_index', 
    'sqale_debt_ratio', 'sqale_rating', 'reliability_rating', 'security_rating', 'security_review_rating']

features = np.nan_to_num(data[features_labels].to_numpy())
target = data['answer'].to_numpy()

target_xboost = [x-1 for x in target]
target = target_xboost


repetitions = 5
data = []
for i in range(repetitions):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(features, target, test_size=0.2, random_state=i)
    features_train, target_train = RandomOverSampler(random_state=0).fit_resample(features_train, target_train)
    row = {
        'features_train': features_train,
        'features_test': features_test,
        'target_train': target_train,
        'target_test': target_test
    }
    data.append(row)


predictores = []


###### DUMMY ######
model = TwoLayersClassifier(classifier=dummy.DummyClassifier)
params = {
    'classifier': [dummy.DummyClassifier],
    'strategy': ['most_frequent', 'prior', 'stratified', 'uniform', 'constant'],
    'constant': [0, 1]
}
predictores.append({ 'name': 'DummyClassifier', 'model': model, 'params': params })

##### NAIVE BAYES #####
model = TwoLayersClassifier(classifier=naive_bayes.GaussianNB)
params = {
    'classifier': [naive_bayes.GaussianNB],
    'var_smoothing': np.logspace(0,-9, num=100)
}
predictores.append({ 'name': 'GaussianNB', 'model': model, 'params': params })

# ###### NEIGHBORS ######
model = TwoLayersClassifier(classifier=neighbors.KNeighborsClassifier)
params = {
    'classifier': [neighbors.KNeighborsClassifier],
    'n_neighbors': [int(x) for x in np.linspace(1, 20, num=20)],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [int(x) for x in np.linspace(1, 40, num=20)]
}
predictores.append({ 'name': 'KNeighborsClassifier', 'model': model, 'params': params })


###### LOGISTIC REGRESSION ######
model = TwoLayersClassifier(classifier=linear_model.LogisticRegression)
params = {
    'classifier': [linear_model.LogisticRegression],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    "C": [uniform.rvs(0.01, 100) for i in range(0, 10)]
}
predictores.append({ 'name': 'LogisticRegression', 'model': model, 'params': params })

###### LINEAR ######
model = TwoLayersClassifier(classifier=linear_model.RidgeClassifier)
params = {
    'classifier': [linear_model.RidgeClassifier],
    'alpha': [uniform.rvs(0.01, 100) for i in range(0, 10)],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
}
predictores.append({ 'name': 'RidgeClassifier', 'model': model, 'params': params })


##### SVM ######
model = TwoLayersClassifier(classifier=svm.SVC)
params = {
    'classifier': [svm.SVC],
    "C": [uniform.rvs(0.01, 10) for i in range(0, 10)],
    'kernel': ['linear']
}
predictores.append({ 'name': 'SVC', 'model': model, 'params': params })

###### TREES ######
model = TwoLayersClassifier(classifier=tree.DecisionTreeClassifier)
params = {
    'classifier': [tree.DecisionTreeClassifier],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'min_samples_leaf': [int(x) for x in np.linspace(1, 11, num = 11)],
    'max_features': ['auto', 'sqrt', 'log2'],
    'splitter': ['best', 'random'],
    'min_weight_fraction_leaf': [0, 1, 2, 4, 8],    
    'max_leaf_nodes': [None, 20, 100, 500, 1000],
}
predictores.append({ 'name': 'DecisionTreeClassifier', 'model': model, 'params': params })


###### RANDOM FOREST ######
model = TwoLayersClassifier(classifier=ensemble.RandomForestClassifier)
params = {
    'classifier': [ensemble.RandomForestClassifier],
    'n_estimators': [int(x) for x in np.linspace(10, 100, num = 18)],
    'criterion': ['gini', 'entropy', 'log_loss'],    
    'min_samples_split': [int(x) for x in np.linspace(2, 11, num = 10)],
    'min_samples_leaf': [int(x) for x in np.linspace(1, 11, num = 11)],
    'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
    'max_features': [int(x) for x in np.linspace(1, 64, num = 16)],    
    'min_weight_fraction_leaf': [0, 1, 2, 4, 8, 16]
}
predictores.append({ 'name': 'RandomForestClassifier', 'model': model, 'params': params })


###### XGBOOST ######
model = TwoLayersClassifier(classifier=xgb.XGBClassifier)
params = {
    'classifier': [xgb.XGBClassifier],
    'objective': ['binary:logistic'],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'n_estimators': [int(x) for x in np.linspace(10, 200, num = 20)],    
    'max_depth': [int(x) for x in np.linspace(5, 50, num = 10)],
    'learning_rate': [uniform.rvs(0.01, 1) for i in range(0, 10)],
    'subsample': [uniform.rvs(0.01, 1) for i in range(0, 10)],
    'colsample_bytree': [uniform.rvs(0.01, 1) for i in range(0, 10)]
}
predictores.append({ 'name': 'XGBOOST', 'model': model, 'params': params })



best_score = 0
for predictor in predictores:
    model = predictor['model']
    params = predictor['params']

    combinations = 100 * len(params)

    for i in range(repetitions):
        row = data[i]
        features_train = row['features_train']
        target_train = row['target_train']

        fold_count=5
        kf = KFold(n_splits=fold_count, shuffle=True)

        clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring='accuracy', n_jobs=-1, cv=kf, verbose=10)
        search = clf.fit(features_train, target_train)

        if i == 0:
            predictor['best_score'] = []
            predictor['best_params'] = []

        predictor['best_score'].append(search.best_score_)
        predictor['best_params'].append(search.best_params_)

        if search.best_score_ > best_score:
            best_score = search.best_score_

        with open('output/results.out', 'a') as f:
            f.write("Running training " + str(i) + ": " + predictor['name'] + '\n\n')
    
    predictor['best_score_mean'] = np.mean(predictor['best_score'])
    print(predictor['best_score'])
    print(predictor['best_score_mean'])

    with open('output/results.out', 'a') as f:
        f.write("Training \n")
        f.write("Predictor: " + predictor['name'] + '\n')
        f.write("Best Score: " + str(predictor['best_score']) + '\n')
        f.write("Best Params: " + str(predictor['best_score_mean']) + '\n')
        f.write('\n\n')




#### Test ####
def test_method(model, params):
    model.set_params(**params)
    target_predicted = model.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, target_predicted)
    precision = precision_score(target_test, target_predicted, average='macro')
    recall = recall_score(target_test, target_predicted, average='macro')
    f1 = f1_score(target_test, target_predicted, average='macro')
    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1
    }

print("\nTESTS\n")
for predictor in predictores:
    name = predictor['name']
    model = predictor['model']
    
    accuracies_test = []
    precisions_test = []
    recalls_test = []
    f1s_test = []

    for i in range(repetitions):        
        params = predictor['best_params'][i]

        with open('output/results.out', 'a') as f:
            f.write("Testing " + str(i) + ": " + name + "\n")
        print(i, name, params)
        test_results = test_method(model, params)
        # print(test_results)
        accuracies_test.append(test_results['accuracy'])
        precisions_test.append(test_results['precision'])
        recalls_test.append(test_results['recall'])
        f1s_test.append(test_results['f1'])

    accuracy_test_mean = np.mean(accuracies_test)
    precision_test_mean = np.mean(precisions_test)
    recall_test_mean = np.mean(recalls_test)
    f1_test_mean = np.mean(f1s_test)

    with open('output/results.out', 'a') as f:
        f.write("\nTest \n")
        f.write("Predictor: " + name + '\n')
        f.write("Accuracies: " + str(accuracies_test) + '\n')
        f.write("Accuracy mean: " + str(accuracy_test_mean) + '\n')
        f.write("Precisions: " + str(precisions_test) + '\n')
        f.write("Precision mean: " + str(precision_test_mean) + '\n')
        f.write("Recalls: " + str(recalls_test) + '\n')
        f.write("Recall mean: " + str(recall_test_mean) + '\n')
        f.write("F1s: " + str(f1s_test) + '\n')
        f.write("F1 mean: " + str(f1_test_mean) + '\n')
        f.write('\n\n')

    print("Method", name)

    print('Accuracy test')
    print(accuracies_test)
    print(accuracy_test_mean)

    print("Precision test")
    print(precisions_test)
    print(precision_test_mean)

    print("Recall test")
    print(recalls_test)
    print(recall_test_mean)

    print("F1 test")
    print(f1s_test)
    print(f1_test_mean)

    print()