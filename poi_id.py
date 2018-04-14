#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("./tools/")

from sklearn.cross_validation import train_test_split
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for k, v in zip(data_dict.keys(), data_dict.values()):
	to_proportion = float(v["from_poi_to_this_person"]) / float(v["to_messages"])
	from_proportion = float(v["from_this_person_to_poi"]) / float(v["from_messages"])
	suspicious_email_weight = (to_proportion + from_proportion) / 2
	if np.isnan(suspicious_email_weight):
		#assign numerical value to missing items, since DT needs a value
		suspicious_email_weight = -1
	data_dict[k]["suspicious_email_weight"] = suspicious_email_weight

my_dataset = data_dict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
df = pd.DataFrame.from_dict(data_dict, orient="index")

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'suspicious_email_weight']

### Task 2: Remove outliers
#total row is an unwanted outlier; 
#other outliers may be poi's, so I will not remove individuals from the data
del data_dict['TOTAL']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

pca = PCA(n_components=2)
selection = SelectKBest(k=1)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
combined_features.fit(features, labels).transform(features)

DT = DecisionTreeClassifier(min_samples_split=6, random_state=42)

pipe = Pipeline([("features", combined_features), ("DT", DT)])
param_grid = dict(
	features__pca__n_components=[1,2,3],
	features__univ_select__score_func=[f_classif, mutual_info_classif],
	features__univ_select__k=[2,3,4],
	DT__min_samples_split=[2,4,6,8,10],
	DT__max_depth=[5,10,20,40]
	)
grid = GridSearchCV(pipe, param_grid=param_grid, scoring='recall_micro')
grid.fit(features, labels)
pd.DataFrame(grid.cv_results_).to_csv('grid_results.csv')
clf = grid.best_estimator_
print clf

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

##### Note from student: tuning of parameters done above along with classifier selection
##### Also, it looks like the tester.py file is using StratifiedShuffleSplit. 
##### I have been using tester.py to evaluate my algorithms thus far. 

# Example starting point. Try investigating other evaluation techniques!
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)