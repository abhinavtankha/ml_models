# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# Feature Selection Tactics - https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# code for adabooster with concept - https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()

target_column = ['Outcome'] 

X = df.iloc[:, 0:7]
Y = df.iloc[:, 8]

train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=40)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

parameter_space = {
    'n_estimators': [10, 30, 70, 100, 150],
    'learning_rate': [ 0.1, 0.3, 0.5]
}

oversample = SMOTE(random_state=40)
train_X, train_y = oversample.fit_resample(train_X, train_y)

sc_X = MinMaxScaler(feature_range = (0, 1))
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_split=30, random_state=40), random_state=40)

clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_X, train_y)

print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
scores = []
count = 0
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    scores.append(mean)
    count = count+1

print("Report")

predict_train = clf.predict(train_X)
predict_test = clf.predict(test_X)

print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

print('Accuracy is {}'.format(accuracy_score(test_y,predict_test)))


#print(confusion_matrix(test_y, predictions))


# [[269   0]
#  [  0 157]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00       269
#            1       1.00      1.00      1.00       157

#     accuracy                           1.00       426
#    macro avg       1.00      1.00      1.00       426
# weighted avg       1.00      1.00      1.00       426

# [[84  4]
#  [ 9 46]]
#               precision    recall  f1-score   support

#            0       0.90      0.95      0.93        88
#            1       0.92      0.84      0.88        55

#     accuracy                           0.91       143
#    macro avg       0.91      0.90      0.90       143
# weighted avg       0.91      0.91      0.91       143
