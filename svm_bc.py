# code reference - https://towardsdatascience.com/support-vector-machine-mnist-digit-classification-with-python-including-my-hand-written-digits-83d6eca7004a

import math, time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

data = pd.read_csv("bcancer.csv")

data.replace({"diagnosis": {"M": 1, "B": 0}}, inplace=True) # replace the encoding with 0 and 1

# sel_col = np.array([0, 1, 2, 3, 12, 13, 20, 21, 22, 23]) + 2
sel_col = np.array([27, 22, 7, 20, 2, 23, 0, 3, 6, 26]) + 2

X = data.iloc[:,sel_col.tolist()]
Y = data.iloc[:,1]
print(Y.value_counts())
train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=40, train_size=0.3)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

# # Oversampling(if needed)
# oversample = SMOTE(random_state=1)
# train_X, train_y = oversample.fit_resample(train_X, train_y)

steps = [('scaler', StandardScaler()), ('SVM', SVC(random_state=40))]
pipeline = Pipeline(steps) # define Pipeline object
parameters = {'SVM__C':[0.1, 10], 'SVM__gamma':[10, 100], 'SVM__kernel': ['poly', 'rbf']}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
grid.fit(train_X, train_y)

print("score = %3.2f" %(grid.score(test_X, test_y)))
print("best parameters from train data: ", grid.best_params_)

means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

predict_train = grid.predict(train_X)
predict_test = grid.predict(test_X)

print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

print('Accuracy is {}'.format(accuracy_score(test_y,predict_test)))

# train_sizes=[30, 50, 70, 90, 110]# for bc
# train_sizes, train_scores, valid_scores = learning_curve(SVC(C = 0.1, gamma=10, kernel='poly'), train_X, train_y, train_sizes=train_sizes, cv = 3)
# print("trainingset size", train_X.shape)
# print(train_scores)
# print(valid_scores)
# print(train_scores[:, 0])
# print(valid_scores[:, 0])

# plt.plot(train_sizes,train_scores[:, 2],label='Training')
# plt.plot(train_sizes,valid_scores[:, 2],label='Validation')
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy')
# plt.legend(loc=3,bbox_to_anchor=(1,0))
# plt.show()

# [[115   2]
#  [  2 115]]
#               precision    recall  f1-score   support

#            0       0.98      0.98      0.98       117
#            1       0.98      0.98      0.98       117

#     accuracy                           0.98       234
#    macro avg       0.98      0.98      0.98       234
# weighted avg       0.98      0.98      0.98       234

# [[227  13]
#  [ 14 128]]
#               precision    recall  f1-score   support

#            0       0.94      0.95      0.94       240
#            1       0.91      0.90      0.90       142

#     accuracy                           0.93       382
#    macro avg       0.92      0.92      0.92       382
# weighted avg       0.93      0.93      0.93       382
