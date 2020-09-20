#knn
#reference doc - https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
#knn documentation - https://scikit-learn.org/stable/modules/neighbors.html


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pylab as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("bcancer.csv")

data.replace({"diagnosis": {"M": 1, "B": 0}}, inplace=True) # replace the encoding with 0 and 1

#sel_col = np.array([0, 1, 2, 3, 12, 13, 20, 21, 22, 23]) + 2
sel_col = np.array([27, 22, 7, 20, 2, 23, 0, 3, 6, 26]) + 2

X = data.iloc[:,sel_col.tolist()]
Y = data.iloc[:,1]
train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=1)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

parameter_space = {
    'n_neighbors': range(3,26),
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()

clf = GridSearchCV(knn, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_X, train_y)

# Best paramete set
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

plt.plot(range(1, count + 1), scores)
plt.xlabel('Value of K(# of neighbours)')
plt.ylabel('Score')
plt.show()

train_sizes=[50, 100, 150, 200, 250, 284]# for bc
train_sizes, train_scores, valid_scores = learning_curve(KNeighborsClassifier(n_neighbors= 9, weights='uniform'), train_X, train_y, train_sizes=train_sizes, cv = 3)
print("trainingset size", train_X.shape)
print(train_scores)
print(valid_scores)
print(train_scores[:, 0])
print(valid_scores[:, 0])

plt.plot(train_sizes,train_scores[:, 2],label='Training')
plt.plot(train_sizes,valid_scores[:, 2],label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc=3,bbox_to_anchor=(1,0))
plt.show()

