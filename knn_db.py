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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()

X = df.iloc[:, 0:7]
y = df.iloc[:, 8]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=1)
print(train_X.shape); print(test_X.shape);print(train_y.shape); print(test_y.shape)

sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

parameter_space = {
    'n_neighbors': range(1,26),
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()

clf = GridSearchCV(knn, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_X, train_y)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)
scores = []
count = 0

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
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

train_sizes=[20, 50, 100,200, 300, 340]# for db
train_sizes, train_scores, valid_scores = learning_curve(KNeighborsClassifier(n_neighbors= 15, weights='uniform'), train_X, train_y, train_sizes=train_sizes, cv = 3)
print("trainingset size", train_X.shape)
print(train_scores)
print(valid_scores)
print(train_scores[:, 0])
print(valid_scores[:, 0])

plt.plot(train_sizes,train_scores[:, 0],label='Training')
plt.plot(train_sizes,valid_scores[:, 0],label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc=3,bbox_to_anchor=(1,0))
plt.show()
# Note: 
# With SMOTE getting a accuracy score of 0.7598425196850394
# With SMOTE + MinMaxScaler accuracy score of 0.7125984251968503
# Discarding both
# Number of CV samples have to between [0, 342].. and more than 15 else the number of neighbours is a problem.