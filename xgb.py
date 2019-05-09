#-*-encoding=utf-8-*-

'''
xgb对分类数据使用
'''

import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


path = './data/'
dtrain = xgb.DMatrix(path + 'agaricus.txt.train')
dtest = xgb.DMatrix(path + 'agaricus.txt.test')


param = {
	'max_depth':2,
	'eta':1,
	'slient':0,
	'objective':'binary:logistic'
}

num_round = 4

bst = xgb.train(params=param, dtrain=dtrain,num_boost_round=num_round)


def evaluate(data=dtest, model=bst):
	pre = model.predict(data)
	pre_res = [round(res) for res in pre]
	y = data.get_label()
	acc = accuracy_score(y, pre_res)
	return acc

print(evaluate())

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file

x_train, y_train = load_svmlight_file(path + 'agaricus.txt.train')
x_test, y_test = load_svmlight_file(path + 'agaricus.txt.test')

# x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2)

bst = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, silent=0, objective='binary:logistic')


def evaluate_sklearn(data, model, label):
	pred = model.predict(data)
	pred_res = [round(res) for res in pred]
	acc = accuracy_score(label, pred_res)
	return acc

param_test = {
	'n_estimators': range(1, 50, 1)
}

clf = GridSearchCV(estimator=bst, param_grid=param_test, scoring='accuracy', cv=5)
clf.fit(x_train, y_train)
print(evaluate_sklearn(x_test, clf.best_estimator_, y_test))