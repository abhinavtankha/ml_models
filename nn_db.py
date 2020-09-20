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


df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()

X = df.iloc[:, 0:7]
y = df.iloc[:, 8]

# Actual Param Space
# parameter_space = {
#     'hidden_layer_sizes': [(8,8,8), (20,20,20), (20,), (50,), (50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }


parameter_space = {
    'hidden_layer_sizes': [(6,6,6,6),(7,7,7,7), (9,9,9,9), (8,8,8,8), (10,10,10,10)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [ 0.1, 0.08, 0.04, 0.06],
    'learning_rate': ['constant','adaptive'],
}

# parameter_space = {
#     'hidden_layer_sizes': [(8,8,8)],
#     'activation': ['tanh'],
#     'solver': ['sgd'],
#     'alpha': [0.0001],
#     'learning_rate': ['constant'],
# }

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# sc_X = MinMaxScaler(feature_range = (0, 1))
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

mlp = MLPClassifier(max_iter=5000, random_state=40, early_stopping=False)

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best paramete set
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
#print ("loss values", loss_values)
print ("loss:", clf.best_estimator_.loss_, "iterations:", clf.best_estimator_.n_iter_, "layers", clf.best_estimator_.n_layers_, "number of outputs:", clf.best_estimator_.n_outputs_, "output_act_fn", clf.best_estimator_.out_activation_)
plt.xlabel("Iterations")
plt.ylabel("Loss Value")
plt.title("Loss vs Iterations")
plt.plot(loss_values)
plt.show()



# Observations
# 1. Data is unbalanced. More cases of non patient vs patient. So the results are skewered towards the non-patient. getting a lot of false negatives than false positives. can see in the confusion matrix.
# // why is understandable given the patients are more and the nonpatients are less.
# Result -  {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
# max iterations set to 5000.
# [[323  35]
#  [ 79 100]]
#               precision    recall  f1-score   support

#    Negative      0.80      0.90      0.85       358
#   Positive       0.74      0.56      0.64       179

#     accuracy                           0.79       537
#    macro avg       0.77      0.73      0.74       537
# weighted avg       0.78      0.79      0.78       537

# [[124  18]
#  [ 41  48]]
#               precision    recall  f1-score   support

#   Negative       0.75      0.87      0.81       142
#   Positive       0.73      0.54      0.62        89

#     accuracy                           0.74       231
#    macro avg       0.74      0.71      0.71       231
# weighted avg       0.74      0.74      0.74       231

# After using SMOTE upsampling to get more data for minority class, explain what SMOTE does,
# Observation:
# The True positives have gone down while the true negatives have gone up. Alsso the row for 1 has increased while that for 0 had decreased. This is expected..


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




#17.09 NOtes
# On running the MLP Classifier noticing that the training results are very good, near perfect score, while the test score is very poor.
# So clearly there is overfitting in the model. Sharing the scores below -

# Best Parameters found -  {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'}

# Training Score -

# Confidence Matrix
# [[358   0]
#  [  0 358]]


#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00       358
#            1       1.00      1.00      1.00       358

#     accuracy                           1.00       716
#    macro avg       1.00      1.00      1.00       716
# weighted avg       1.00      1.00      1.00       716


# Testing Score - 

# Confidence Matrix
# [[111  31]
#  [ 42  47]]
#               precision    recall  f1-score   support

#            0       0.73      0.78      0.75       142
#            1       0.60      0.53      0.56        89

#     accuracy                           0.68       231
#    macro avg       0.66      0.65      0.66       231
# weighted avg       0.68      0.68      0.68       231


# To reduce overfitting, 
# 1. Train on more examples - not possible.
# 2. reduce model complexity. increase the regulation penalty for large weights. kept fewer nodes and layers. also added early stopping.
# remember - Perhaps start by testing values on a log scale, such as 0.1, 0.001, and 0.0001. Then use a grid search at the order of magnitude that shows the most promise.
# iterate through number of layers and alpha for penalty term in regularization to reduce overfitting.


# Seeing overfitting. with final test f1-score as (0.72,0.57), while the training f1 score - (0.99, 0.99)
# After Removing (10,10,10, 10) and adding early stopping = True -> (0.74,0.67). The training f1 score has also fallen - 0.73, 0.73. Minor improvement for testing, but training results have fallen. Best parameters - {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (10, 10, 10), 'learning_rate': 'constant', 'solver': 'adam'}
# I think we have entered the underfitting case, so will now revert back the model layers to (10,10,10,10), while keeping the early stopping = false intact. Result : Training 0.78, 0,76, Testing 0.77, 0.68. Again seems to be a case of underfitting. Best Results - {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (10, 10, 10, 10), 'learning_rate': 'constant', 'solver': 'adam'}
# Reversing the 2, keeping the model size as (10,10,10), while making early stopping as false. Result - Training: 0.75, 0.75. Testing 0.75, 0.68. Best param: {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (10, 10, 10), 'learning_rate': 'constant', 'solver': 'adam'}. still seeing underfitting. will have to make the model more complex gradually.
# 'hidden_layer_sizes': [(10),(10, 10), (10,10,10), (5,5,5,5) ], and early stopping = true. Results fell sharply. Training 0.69, 0.58 Testing 0.78, 0.56. {'activation': 'relu', 'alpha': 0.04, 'hidden_layer_sizes': (5, 5, 5, 5), 'learning_rate': 'constant', 'solver': 'adam'}
# early stopping as false, hidden layers the same. Results - Training 0.88, 0.88. Testing 0.76, 0.63. I think we can still work on making the model more accomodative. 
# early stopping as false and 'hidden_layer_sizes': [(6,6,6,6),(7,7,7,7), (9,9,9,9), (8,8,8,8), (10,10,10,10)]. Had to increase the iterations to 5000, as was getting convergence warning.Results? Training 0.79. 0.78 Testing 0.76, 0.68 Best params: {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (10, 10, 10, 10), 'learning_rate': 'constant', 'solver': 'sgd'}
# NOT GETTING A GOOD CONVERGENCE!! BWAH BWAH!!