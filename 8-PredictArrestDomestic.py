#!/usr/bin/python

import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math as ma

from scipy.stats import sem
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def mean_score(scores):
    """Print the empirical mean score and standard error of the mean."""
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def decideArrestDomestic(feature):
    if (feature[0] == True and feature[1] == True):
        return 0
    elif (feature[0] == True and feature[1] == False):
        return 1
    elif (feature[0] == False and feature[1] == True):
        return 2
    else:
        return 3

def transformArrestDomestic(features):
    """ Transform Arrest and Domestic to one feature.
    """
    rows = features.shape[0]
    new_feature = np.zeros(rows)

    i = 0
    for f in features:
        new_feature[i] = decideArrestDomestic(f)
        i += 1

    return new_feature

def fillDistrictNA(dataset):
    column = 4

    i = 0
    for row in dataset:
        if (ma.isnan(row[column])):
            dataset[i, column] = 0;
        i += 1

    return dataset

def loadDataset(csv_file):
    # load without the first line
    raw_dataset = pd.read_csv(csv_file)[1:]
    np_dataset = np.array(raw_dataset)

    arrest_domestic = np_dataset[:, [6, 7]]
    arrest_domestic = transformArrestDomestic(arrest_domestic)
    arrest_domestic = np.transpose(np.asmatrix(arrest_domestic))

    # Block, Primary Type, Description, Location Description, District
    s_dataset = np_dataset[:, [1, 3, 4, 5, 8]]
    s_dataset = fillDistrictNA(s_dataset)

    return np.hstack((arrest_domestic, s_dataset))

def relabel(column):
    le = preprocessing.LabelEncoder()
    le.fit(column)

    return le.transform(column)

def relabelDataset(dataset):
    np_dataset = np.array(dataset)

    target        = relabel(np_dataset[:, 0])
    block         = relabel(np_dataset[:, 1])
    p_type        = relabel(np_dataset[:, 2])
    description   = relabel(np_dataset[:, 3])
    l_description = relabel(np_dataset[:, 4])
    district      = relabel(np_dataset[:, 5])

    target        =  np.transpose(np.asmatrix(target, dtype='float'))
    block         =  np.transpose(np.asmatrix(block, dtype='float'))
    p_type        =  np.transpose(np.asmatrix(p_type, dtype='float'))
    description   =  np.transpose(np.asmatrix(description, dtype='float'))
    l_description =  np.transpose(np.asmatrix(l_description, dtype='float'))
    district      =  np.transpose(np.asmatrix(district, dtype='float'))

    new_dataset = np.hstack((target, block))
    new_dataset = np.hstack((new_dataset, p_type))
    new_dataset = np.hstack((new_dataset, description))
    new_dataset = np.hstack((new_dataset, l_description))
    new_dataset = np.hstack((new_dataset, district))

    return new_dataset

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)

    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)

    print mean_score(scores)

def main():
    if (len(sys.argv) > 1):
        dataset_file = sys.argv[1]
    else:
        print "You have not specified the name of input dataset."
        exit(1)

    crimes = loadDataset(dataset_file)
    crimes = relabelDataset(crimes)

    X_crimes, y_crimes = crimes[:,1:], crimes[:, 0]
    y_crimes = np.ravel(y_crimes) #return a flattened array.

    # create a composite estimator made by a pipeline of the standarization and the linear model
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', RandomForestClassifier())
        #('svm', SVC(kernel='linear'))
        ])

    evaluate_cross_validation(clf, X_crimes, y_crimes, 5)

if __name__ == "__main__":
    main()
