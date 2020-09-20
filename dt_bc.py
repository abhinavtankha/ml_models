import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV
import pydotplus
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

data = pd.read_csv("bcancer.csv")


data.replace({"diagnosis": {"M": 1, "B": 0}}, inplace=True) # replace the encoding with 0 and 1

#sel_col = np.array([0, 1, 2, 3, 12, 13, 20, 21, 22, 23]) + 2
sel_col = np.array([27, 22, 7, 20, 2, 23, 0, 3, 6, 26]) + 2

labels = data.columns.values[sel_col]
print(labels)

X = data.iloc[:,sel_col.tolist()]
Y = data.iloc[:,1]
print("value counts", Y.value_counts())

train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=40)
print("Train and Test Sizes", train_X.shape, test_X.shape, train_y.shape, test_y.shape)


oversample = SMOTE(random_state=40)
train_X, train_y = oversample.fit_resample(train_X, train_y)

print("value counts after upsampling", train_y.value_counts())

# StandardScaler
# sc_X = StandardScaler()
# train_X = sc_X.fit_transform(train_X)
# test_X = sc_X.transform(test_X)

# MinMaxScaler
sc_X = MinMaxScaler(feature_range = (0, 1))
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

classifier = DecisionTreeClassifier(criterion = 'entropy',random_state=40)
classifier = classifier.fit(train_X,train_y)

print("depth of the tree: ", classifier.get_depth())

predict_train = classifier.predict(train_X)
predict_test = classifier.predict(test_X)

print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

print("Params before pruning:", classifier.get_params())

## Plain Text Visual Representation
# text_representation = tree.export_text(classifier, feature_names=labels.tolist())
# print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier, fontsize = 14,
                   feature_names=labels.tolist(),  
                   #class_names=["Cancer"],
                   filled=True)
fig.savefig("decistion_tree.png")
print('Accuracy is {}'.format(accuracy_score(test_y,predict_test)))

parameter_space = {
    'max_depth': range(1,5),
    'criterion': ['entropy'],
    'min_samples_split' : [5, 10, 20, 30]
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

## Plain Text Visual Representation
#classifier = classifier.fit(train_X,train_y)#Predict the response for test dataset
# print("depth of the tree, after pruning: ", clf.get_depth())
predict_train = clf.predict(train_X)
predict_test = clf.predict(test_X)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("### AFTER PRUNING ###")
print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

text_representation = tree.export_text(clf.best_estimator_, feature_names=labels.tolist())
print(text_representation)

print("Params for after pruning:", clf.best_estimator_.get_params())
print('Accuracy is {}'.format(accuracy_score(test_y,predict_test)))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf.best_estimator_, fontsize = 18,
                   feature_names=labels.tolist(),  
                   filled=True)
fig.savefig("decision_tree_bc_pruning.png")
plt.clf()

train_sizes=[50, 100, 150, 200, 250, 284]# for bc
train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(criterion='entropy' , max_depth=3, min_samples_split=5), train_X, train_y, train_sizes=train_sizes, cv = 3)
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

#Notes
# 1. get the pruning working through a GridCV, max depth field esp
# 2. get the visualization working.
# 3. get the GridCV working.

# 1. Ran a CV to get the improvement with param tuning
# parameter_space = {
#     'max_depth': range(1,5),
#     'criterion': ['entropy', 'gini'],
#     'min_samples_split' : [ 20, 50, 80, 120]
# } Results- Tr(98, 98), Te(91, 85),  {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 20}

#2. I tried to increase to make leaves more coarser to reduce overfitting but after a size of 30 I started noticing a fall in performance. So fixed that.
#3. criterion enntropy was performing better than gini.
#4. max depth of 4 was best. still at Tr(98, 98), Te(91, 85) - {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 30}
#5 final result
# parameter_space = {
#     'max_depth': range(1,10),
#     'criterion': ['entropy'],
#     'min_samples_split' : [30,40]
# }



