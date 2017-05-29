### Test to see if balancing-tree training data makes for more useful feature selection

# Suppress package warnings, most of which are caused by deprecations
import warnings
warnings.filterwarnings('ignore')

### Load the dictionary containing financial data and some email features (to_ from_poi)
import pickle
with open('data/final_project_dataset.pkl', 'rb') as f:
    fin_data_dict = pickle.load(f)

# Remove outliers
fin_data_dict.pop('TOTAL', 0)

# Store to my_dataset for easy export below.
my_dataset = fin_data_dict

### Select features to use

# features_list is a list of feature names in the financial data.
# The first feature must be "poi".

import numpy as np
from tools.feature_format import featureFormat, targetFeatureSplit

def make_features_labels(dataset, feature_names):
    """
    Quick way to split a dataset into features and labels based on feature names
    """
    data = featureFormat(dataset, feature_names, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    labels = np.array(labels)
    features = np.array(features)
    
    return features, labels

# Start with all features except: 'email_address'

all_feature_names = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments','exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi','restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances','from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income','long_term_incentive', 'from_poi_to_this_person']

all_features, all_labels = make_features_labels(my_dataset, all_feature_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, 
                                                    test_size=0.3, random_state=42)

def balancer(X, y):
    balanced_X = []
    balanced_y = []
    count = 0
    for i in range(X.shape[0]):
        if y[i] == 1:
            balanced_X.append(X[i, :])
            balanced_y.append(1)
        elif (y[i] == 0 and
              count < sum(y)):
            balanced_X.append(X[i, :])
            balanced_y.append(0)
            count += 1

    return np.array(balanced_X), np.array(balanced_y)

X_balanced, y_balanced = balancer(X_train, y_train)

# # Select the most important features based on ExtraTreesClassifier
from feature_selection import importance_plotter
selected_feature_names = importance_plotter(X_balanced, y_balanced, 
                                            np.array(all_feature_names[1:]))



# # Plot the most highly correlated features to see if there is any redundancy or outliers 
from feature_selection import correlation_plotter
correlation_plotter(selected_feature_names, my_dataset)



# # From the correlation graphs it seems like total_stock_value correlates with a lot
# # of other features and is very similar to exercised_stock_options, so I will remove it.
# selected_feature_names.remove('total_stock_value')



# ### Confirm that the feature selection did help performance:
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

# # Make features and labels based on the new selection
selected_feature_names.insert(0, 'poi')
selected_features, selected_labels = make_features_labels(my_dataset, selected_feature_names)

clf = GaussianNB()
def fit_print_scores(clf, features, labels):

    f1_scores = cross_val_score(clf, features, labels, cv=5, scoring='f1')
    print '-f1 score: %0.2f (+/- %0.2f)' % (f1_scores.mean(), 
                                           f1_scores.std() * 2)
    precision_scores = cross_val_score(clf, features, labels, cv=5, scoring='precision')
    print '-precision score: %0.2f (+/- %0.2f)' % (precision_scores.mean(), 
                                                  precision_scores.std() * 2)
    recall_scores = cross_val_score(clf, features, labels, cv=5, scoring='recall')
    print '-recall score: %0.2f (+/- %0.2f)' % (recall_scores.mean(), 
                                                  recall_scores.std() * 2)

print 'All features:'
fit_print_scores(clf, all_features, all_labels)
print '\nSelected features:'
fit_print_scores(clf, selected_features, selected_labels)
