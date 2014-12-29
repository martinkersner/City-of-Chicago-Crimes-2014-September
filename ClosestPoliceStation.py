#!/usr/bin/python

import pandas as pd
import numpy as np

def LoadPoliceStations(csvFile):
    rawDataset = pd.read_csv(csvFile)
    latitude = np.transpose(np.asmatrix(rawDataset.LATITUDE))
    longitude = np.transpose(np.asmatrix(rawDataset.LONGITUDE))

    return np.hstack((latitude, longitude))

def LoadCrimes(csvFile):
    rawDataset = pd.read_csv(csvFile)
    latitude = np.transpose(np.asmatrix(rawDataset.Latitude))
    longitude = np.transpose(np.asmatrix(rawDataset.Longitude))

    return np.asmatrix(rawDataset), np.hstack((latitude, longitude))

def ComputeClosestPoliceStations(csvFile, policeStations):
    dataset, ll = LoadCrimes(csvFile)
    length = ll.shape[0]

    euclidianColumn = np.zeros((length, 1))
    manhattanColumn = np.zeros((length, 1))

    for i in range(0, length):
        euclidianColumn[i] = ComputeEuclidianDistance(ll[i], policeStations)
        manhattanColumn[i] = ComputeManhattanDistance(ll[i], policeStations)

    tmpDataset = np.hstack((dataset, euclidianColumn))
    return np.hstack((tmpDataset, manhattanColumn))

def ComputeEuclidianDistance(position, policeStations):
    return np.min(np.sqrt(np.sum(np.power((policeStations - position), 2), axis=1)))

def ComputeManhattanDistance(position, policeStations):
    return np.min(np.sum(np.absolute(policeStations - position), axis=1))

def main():
    ps = LoadPoliceStations("data/Police_Stations.csv")
    cps = ComputeClosestPoliceStations("data/Crimes_September_2014.csv", ps)
    df = pd.DataFrame(cps)
    df.to_csv("data/Crimes_September_2014_Euclidian_Manhattan.csv", header = ["ID", "Date", "Block", "Primary Type", "Description", "Location Description", "Arrest", "Domestic", "District", "Latitude", "Longitude", "Euclidian", "Manhattan"])

if __name__ == "__main__":
    main()
