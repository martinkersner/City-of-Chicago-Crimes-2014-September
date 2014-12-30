#!/usr/bin/python

import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import datetime
import math as ma

from scipy.stats import sem
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def mean_score(scores):
    """Print the empirical mean score and standard error of the mean."""
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def date2weekDay(date):
    """ Converts given date(s) to day of the week.
    """
    date_column = np.zeros(shape=(len(date), 1))

    index = 0
    for d in date:
        newrow = weekDay(d)
        date_column[index] = newrow
        index+=1

    return date_column

def weekDay(timeStamp):
    """ Converts date to integer expression of the day of week.
    """
    dt = timeStamp[0:10]
    month, day, year = (int(x) for x in dt.split('/'))

    try:
        ans = datetime.date(year, month, day)
    except:
        year, day, month = (int(x) for x in dt.split('/'))
        ans = datetime.date(year, month, day)

    return day2int(ans.strftime("%A"))

def day2int(day):
    """ Converts day of the week to integer expression.
    """
    if (day == "Monday"):
        return 1
    elif (day == "Tuesday"):
        return 2
    elif (day == "Wednesday"):
        return 3
    elif (day == "Thursday"):
        return 4
    elif (day == "Friday"):
        return 5
    elif (day == "Saturday"):
        return 6
    elif (day == "Sunday"):
        return 7

def loadDataset(csv_file):
    # load without the first line
    raw_dataset = pd.read_csv(csv_file)[1:]
    np_dataset = np.array(raw_dataset)

    date = np_dataset[:, 1]
    day_column = date2weekDay(date)

    # select only 3, 2 and 5 column
    # Primary Type, Block, Location Description, Description
    s_dataset = np_dataset[:, [3, 2, 5, 4]]

    # add day of week
    return np.hstack((s_dataset, day_column))

def relabel(column):
    le = preprocessing.LabelEncoder()
    le.fit(column)
    return le.transform(column)

def relabelDataset(dataset):
    np_dataset = np.array(dataset)

    target      = relabel(np_dataset[:, 0])
    block       = relabel(np_dataset[:, 1])
    location    = relabel(np_dataset[:, 2])
    description = relabel(np_dataset[:, 3])
    day         = relabel(np_dataset[:, 4])

    target      =  np.transpose(np.asmatrix(target, dtype='float'))
    block       =  np.transpose(np.asmatrix(block, dtype='float'))
    location    =  np.transpose(np.asmatrix(location, dtype='float'))
    description =  np.transpose(np.asmatrix(description, dtype='float'))
    day         =  np.transpose(np.asmatrix(day, dtype='float'))

    #new_dataset = np.hstack((target, block))
    new_dataset = np.hstack((target, location))
    new_dataset = np.hstack((new_dataset, description))
    new_dataset = np.hstack((new_dataset, day))

    return new_dataset

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

    # number of classes in each attribute
    #print len(set(np.ravel(crimes[:, 0])))
    #print len(set(np.ravel(crimes[:, 1])))
    #print len(set(np.ravel(crimes[:, 2])))
    #print len(set(np.ravel(crimes[:, 3])))

    # create a composite estimator made by a pipeline of the standarization and the linear model
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', RandomForestClassifier(criterion='gini'))
        ])

    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(crimes.shape[0], 5, shuffle=True, random_state=33)

    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X_crimes, y_crimes, cv=cv)

    print mean_score(scores)

if __name__ == "__main__":
    main()
