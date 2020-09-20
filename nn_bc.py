#Neural Network

#DataSet: https://www.kaggle.com/uciml/pima-indians-diabetes-database

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv("bcancer.csv")

data.replace({"diagnosis": {"M": 1, "B": 0}}, inplace=True) # replace the encoding with 0 and 1

#sel_col = np.array([0, 1, 2, 3, 12, 13, 20, 21, 22, 23]) + 2 #chi2 best 10 param
sel_col = np.array([27, 22, 7, 20, 2, 23, 0, 3, 6, 26]) + 2 # ANOVA best 10 param

X = data.iloc[:,sel_col.tolist()]
y = data.iloc[:,1]
print("value counts", y.value_counts())

# Actual Param Space
# parameter_space = {
#     'hidden_layer_sizes': [(8,8,8,8)],
#     'activation': ['relu'],
#     'solver': ['adam'],
#     'alpha': [ 0.1, 0.01, 0.001, 0.0001],
#     'learning_rate': ['constant','adaptive'],
#     'beta_1': [0.9],
#     'beta_2':[0.3, 0.35, 0.4, 0.45, 0.5]
# }

# Shortened Param Space
parameter_space = {
    'hidden_layer_sizes': [(8,8,8,8)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [ 0.01],
    'learning_rate': ['constant','adaptive']
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

# sc_X = MinMaxScaler(feature_range = (0, 1))
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

mlp = MLPClassifier(max_iter=3000, random_state=40, early_stopping= False)

oversample = SMOTE(random_state=40)

X_train, y_train = oversample.fit_resample(X_train, y_train)

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best parameters set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


print("Report")
predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)

print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
print('Accuracy is {}'.format(accuracy_score(y_test,predict_test)))


loss_values = clf.best_estimator_.loss_curve_
print ("loss:", clf.best_estimator_.loss_, "iterations:", clf.best_estimator_.n_iter_, "layers", clf.best_estimator_.n_layers_, "number of outputs:", clf.best_estimator_.n_outputs_, "output_act_fn", clf.best_estimator_.out_activation_)
plt.plot(loss_values)
plt.xlabel("Iterations")
plt.ylabel("Loss Value")
plt.title("Loss vs Iterations")
plt.show()

# Observations
# 

# TODOs:
# 1. How can i modify the data to make it more balanced.
# 2. look into ROC Curves and Kappa for classification accuracy.
# 3. Try oversampling and undersampling of data.
#   Consider testing under-sampling when you have an a lot data (tens- or hundreds of thousands of instances or more)
#   Consider testing over-sampling when you don’t have a lot of data (tens of thousands of records or less)
# 4. Try "SMOTE or the Synthetic Minority Over-sampling Technique". Chack for UnbalancedDataSet module. it provides impementation of SMOTE.
# 5. Remember Decision Trees perform well on imbalanced data sets. // reason why data set is interesting.
# Try few of the populate decision tree algorithms c4.5, c5.0, CART and Random Forests.
# 6. Try penalized models - penalized-SVM and penalized-LDA.,  CostSensitiveClassifier
# Read more about unbalanced dataset in quora.


#17.09 Notes

# To reduce overfitting, 
# 1. Train on more examples
# 2. reduce model complexity. increase the regulation penalty for large weights. kept fewer nodes and layers. also added early stopping.
# remember - Perhaps start by testing values on a log scale, such as 0.1, 0.001, and 0.0001. Then use a grid search at the order of magnitude that shows the most promise.
# iterate through number of layers and alpha for penalty term in regularization.


# Execution 1:
# No early stopping and still with a test set of layers and regularization term.

# {'activation': 'relu', 'alpha': 0.08, 'hidden_layer_sizes': (10, 10, 10, 10), 'learning_rate': 'constant', 'solver': 'adam'}

# Report
# [[241   1]
#  [  1 241]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00       242
#            1       1.00      1.00      1.00       242

#     accuracy                           1.00       484
#    macro avg       1.00      1.00      1.00       484
# weighted avg       1.00      1.00      1.00       484

# [[109   6]
#  [  3  53]]
#               precision    recall  f1-score   support

#            0       0.97      0.95      0.96       115
#            1       0.90      0.95      0.92        56

#     accuracy                           0.95       171
#    macro avg       0.94      0.95      0.94       171
# weighted avg       0.95      0.95      0.95       171

# loss: 0.02790397727163788 iterations: 729 layers 6 number of outputs: 1 output_act_fn logistic

# Performance is good. Already in the range of 92% f1 score.

# Now need to figure out the right # of layers and also the regularization term.
# I notice some overfitting in the training set performance. The data was overlearnt to 100% accuracy - so seems to be a case of overfitting. 
# Next Step - 
# I will try to make teh model less complex 
# 	1. Reduce the layers and nodes
# 	2. Increase the range of regularization penalty term.
# 	3. Add early stopping.


# Added early stopping = True, the f1 score for training dropped to (0.94, 0.94) and test results f1 score dropped to (0.92, 0.86). Need to keep early stopping as false for now.
# Keeping the early stopping as False but expanding variability in 'alpha': [ 0.1, 0.01, 0.001, 0.0001] for the tests. - Result - Training 1.00, 1.00, Testing 0.96, 0.93. best params - {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (10, 10, 10, 10), 'learning_rate': 'constant', 'solver': 'adam'}
# Reducing 'hidden_layer_sizes': [(10),(10, 10), (10,10,10) ]. Result - Training 1.00, 1.00. Testing 0.97, 0.94. Best param : {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (10, 10, 10), 'learning_rate': 'constant', 'solver': 'adam'}. Results are best so far. 
# Adding a hidden layer - 'hidden_layer_sizes': [(10),(10, 10), (8,8,8) ], while keeping the early stopping as false. Results. Training 0.99, 0.99. Testing 0.97, 0.94.  Best params:  {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (8, 8, 8), 'learning_rate': 'constant', 'solver': 'adam'}. Not much improvement.
# will try to increase another hidden layer and make it small - 'hidden_layer_sizes': [(8,8,8), (2,2,2,2), (5,5,5,5), (8,8,8,8) ]. Results. Training 0.99, 0.99. Testing 0.98, 0.96. Best Param:  {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (8, 8, 8, 8), 'learning_rate': 'constant', 'solver': 'adam'}
# Fixing most of the parameters and trying for different adam parameters, beta1 and beta2
# parameter_space = {
#     'hidden_layer_sizes': [(8,8,8,8)],
#     'activation': ['relu'],
#     'solver': ['adam'],
#     'alpha': [ 0.1, 0.01, 0.001, 0.0001],
#     'learning_rate': ['constant','adaptive'],
#     'beta_1': [0.1, 0.3, 0.5, 0.7, 0.9],
#     'beta_2':[0.1, 0.3, 0.5, 0.7, 0.9]
# } Result. Not as good. Training 0.99,0.99 Testing 0.96,0.91. Best param:  {'activation': 'relu', 'alpha': 0.1, 'beta_1': 0.9, 'beta_2': 0.3, 'hidden_layer_sizes': (8, 8, 8, 8), 'learning_rate': 'constant', 'solver': 'adam'}


#Fixing beta1 as default 0.9. 'beta_1': [0.9],'beta_2':[0.3, 0.35, 0.4, 0.45, 0.5]. Results?

# Final Score
# Best Params -  {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (8, 8, 8, 8), 'learning_rate': 'constant', 'solver': 'adam'}
# [[241   1]
#  [  0 242]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00       242
#            1       1.00      1.00      1.00       242

#     accuracy                           1.00       484
#    macro avg       1.00      1.00      1.00       484
# weighted avg       1.00      1.00      1.00       484

# [[111   4]
#  [  2  54]]
#               precision    recall  f1-score   support

#            0       0.98      0.97      0.97       115
#            1       0.93      0.96      0.95        56

#     accuracy                           0.96       171
#    macro avg       0.96      0.96      0.96       171
# weighted avg       0.97      0.96      0.97       171
