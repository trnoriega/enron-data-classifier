import numpy as np
import matplotlib.pyplot as plt

# For importance_plotter
import matplotlib.patches as mpatches
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# For correlation_plotter
from itertools import combinations
from sklearn.linear_model import LinearRegression
from tools.feature_format import featureFormat, targetFeatureSplit

###
# import pickle

# with open('data/final_project_dataset.pkl', 'rb') as f:
#     fin_data_dict = pickle.load(f)

# # Remove outliers
# fin_data_dict.pop('TOTAL', 0)

# # Store to my_dataset for easy export below.
# my_dataset = fin_data_dict

# feature_names = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',\
# 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',\
# 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',\
# 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',\
# 'long_term_incentive', 'from_poi_to_this_person']

# import numpy as np
# from tools.feature_format import featureFormat, targetFeatureSplit
# data = featureFormat(my_dataset, feature_names, sort_keys = True)
# labels, features = targetFeatureSplit(data)
# labels = np.array(labels)
# features = np.array(features)
###

def importance_plotter(features, labels, feature_names):
    """
    Makes a bar graph of feature importances based on ExtraTreesClassifier.
    Returns a list with the features selected to be above the importance mean. 
    """
    forest = ExtraTreesClassifier(n_estimators=500,
                                  random_state=0)
    forest.fit(features, labels)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    name_array = np.array(feature_names)
    sorted_names = name_array[indices]

    model = SelectFromModel(forest, prefit=True, threshold='mean')
    selected_indices = model.get_support()
    selected_names = name_array[selected_indices]

    # Plot the feature importances and which were selected
    plt.close('all')
    fig = plt.figure(1, figsize=(10, 7), tight_layout=True)
    ax = fig.gca(title='Feature selection based on ExtraTreesClassifier importances',
                 xlabel='Feature name',
                 xlim=[-1, features.shape[1]],
                 xticks=range(len(sorted_names)),
                 ylabel='Importance')
    bar_list = ax.bar(range(features.shape[1]), importances[indices],
                      color='b', yerr=std[indices], align='center')
    ax.set_xticklabels(sorted_names, rotation='45', ha='right')
    # color selected features red
    for i, name in enumerate(sorted_names):
        if name in selected_names:
            bar_list[i].set_color('r')
    # create custom legend
    red_patch = mpatches.Patch(color='red', label='Features selected')
    blue_patch = mpatches.Patch(color='blue', label='Features not selected')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    return list(selected_names)

# selected_names = importance_plotter(features, labels, feature_names[1:])

def correlation_plotter(selected_names, my_dataset):
    reg = LinearRegression()
    scores = []

    for pair in combinations(selected_names, 2):
        pair_data = featureFormat(my_dataset, [pair[0], pair[1]])
        d1 = np.reshape(pair_data[:, 0], (len(pair_data[:, 0]), 1))
        d2 = np.reshape(pair_data[:, 1], (len(pair_data[:, 1]), 1))
        reg.fit(d1, d2)
        score = reg.score(d1, d2)
        scores.append((pair[0], pair[1], d1, d2, score))

    scores = sorted(scores, key=lambda score: score[4], reverse=True)
    top_scores = scores[:6]
    fig = plt.figure(2, figsize=(10, 7), tight_layout=True)
    for i, score in enumerate(top_scores):
        X = score[2]
        y = score[3]
        ax = fig.add_subplot(231+i)
        ax.scatter(X, y)
        ax.set_xlabel(score[0])
        ax.set_ylabel(score[1])
        title = 'R2: ' + str(round(score[4], 3))
        ax.set_title(title)
    plt.show()

# scores = correlation_plotter(selected_names, my_dataset)
