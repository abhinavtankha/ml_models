# decision trees tend to overfit a lot. 2 ways to reduce overfitting. 1. pruning and random forests.

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve

df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()

labels = df.columns.values[0:8]
print(labels)

X = df.iloc[:, 0:7]
Y = df.iloc[:, 8]

train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=1)
print("Train and Test Sizes", train_X.shape, test_X.shape, train_y.shape, test_y.shape)

# # Oversampling(if needed)
# oversample = SMOTE(random_state=1)

# train_X, train_y = oversample.fit_resample(train_X, train_y)

# Feature Scaling
sc_X = StandardScaler()
# sc_X = MinMaxScaler(feature_range = (0, 1))
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

classifier = DecisionTreeClassifier(random_state=40)
classifier = classifier.fit(train_X,train_y)
print("depth of the tree: ", classifier.get_depth())
print("Params before pruning:", classifier.get_params())

#prediction
predict_train = classifier.predict(train_X)
predict_test = classifier.predict(test_X)

print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier, fontsize = 14,
                   feature_names=labels.tolist(),  
                   filled=True)
fig.savefig("decision_db_tree.png")
print('Accuracy is {}'.format(accuracy_score(test_y,predict_test)))

parameter_space = {
    'max_depth': range(1,10),
    'criterion': ['entropy'],
    'min_samples_split' : [30,40]
}

# Create Decision Tree classifer object
classifier = DecisionTreeClassifier(criterion="entropy", random_state=40)# Train Decision Tree Classifer

clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_X, train_y)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

predict_train = clf.predict(train_X)
predict_test = clf.predict(test_X)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("### AFTER PRUNING ###")
print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

print('Accuracy is {}'.format(accuracy_score(test_y,predict_test)))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf.best_estimator_, fontsize = 18,
                   feature_names=labels.tolist(),  
                   filled=True)
fig.savefig("decision_tree_db_pruning.png")
plt.clf()

train_sizes=[20, 50, 100,200, 300, 340]# for db
train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(criterion='entropy' , max_depth=2, min_samples_split=30), train_X, train_y, train_sizes=train_sizes, cv = 3)
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

#Notes
# 1. get the pruning working through a GridCV, max depth field esp
# 2. get the visualization working.
# 3. get the GridCV working.

