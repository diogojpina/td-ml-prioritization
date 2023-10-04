import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from scipy.stats import uniform
from sklearn import model_selection
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
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
model = dummy.DummyClassifier()
params = {
    'strategy': ['most_frequent', 'prior', 'stratified', 'uniform', 'constant'],
    'constant': [0, 1]
}
predictores.append({ 'name': 'DummyClassifier', 'model': model, 'params': params })

##### NAIVE BAYES #####
model = naive_bayes.GaussianNB()
params = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
predictores.append({ 'name': 'GaussianNB', 'model': model, 'params': params })

# ###### NEIGHBORS ######
model = neighbors.KNeighborsClassifier()
params = {
    'n_neighbors': [int(x) for x in np.linspace(1, 20, num=20)],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [int(x) for x in np.linspace(1, 40, num=20)]
}
predictores.append({ 'name': 'KNeighborsClassifier', 'model': model, 'params': params })

###### LOGISTIC REGRESSION ######
model = linear_model.LogisticRegression()
params = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    "C": [uniform.rvs(0.01, 100) for i in range(0, 10)]
}
predictores.append({ 'name': 'LogisticRegression', 'model': model, 'params': params })

###### LINEAR ######
model = linear_model.RidgeClassifier()
params = {
    'alpha': [uniform.rvs(0.01, 100) for i in range(0, 10)],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
}
predictores.append({ 'name': 'RidgeClassifier', 'model': model, 'params': params })


##### SVM ######
model = svm.SVC()
params = {
    "C": [uniform.rvs(0.01, 10) for i in range(0, 10)],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
predictores.append({ 'name': 'SVC', 'model': model, 'params': params })

###### TREES ######
model = tree.DecisionTreeClassifier()
params = {
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
model = ensemble.RandomForestClassifier()
params = {
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
model = xgb.XGBClassifier()
params = {
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

        scorer_tuned = make_scorer(accuracy_tuned)
        clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=combinations, scoring=scorer_tuned, n_jobs=-1, cv=kf, verbose=10)
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

    labels = [0, 1, 2, 3, 4, 5]

    accuracy = accuracy_tuned(target_test, target_predicted)
    precision = precision_tuned(target_test, target_predicted, labels)
    recall = recall_tuned(target_test, target_predicted, labels)
    f1 = f1_tuned(target_test, target_predicted, labels)
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