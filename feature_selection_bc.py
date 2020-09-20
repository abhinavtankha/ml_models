# Feature Selection Code

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

data = pd.read_csv("bcancer.csv")

X = data.iloc[:,2:32]
Y = data.iloc[:,1]

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k = 10)
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat the 2 dataframes
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Features', 'Scores']
a = featureScores.nlargest(10, 'Scores')
print(type(a.sort_index()))
print(a)

# # Output of Feature Selection 
#                 Features      Scores
# 27  concave points_worst  964.385393
# 22       perimeter_worst  897.944219
# 7    concave points_mean  861.676020
# 20          radius_worst  860.781707
# 2         perimeter_mean  697.235272
# 23            area_worst  661.600206
# 0            radius_mean  646.981021
# 3              area_mean  573.060747
# 6         concavity_mean  533.793126
# 26       concavity_worst  436.691939f