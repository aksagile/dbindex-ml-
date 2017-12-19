# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:01:34 2017

@author: USER
"""

import pandas as pd
import math
from sklearn.cluster import k_means
from scipy.spatial import distance


df = pd.read_csv('Dataset_Round1_Assignment 1 (of 2)_Programmer.csv')
df = df.dropna()

X1 = df.copy()
del X1['Customer']
del X1['Effective To Date']
X4 = pd.get_dummies(X1)
n=10

clf = k_means(X4,n_clusters=n)

centroids = clf[0]
labels = clf[1]



def compute_s(i,x,centroids,labels,nc):
    norm = math.sqrt(nc)
    s = 0
    for x in centroids:
        s+= distance.euclidean(x,centroids[i])
    s= s/norm
    return s
    
def compute_Rij(i,j,x,centroids,labels,nc):
    dist = distance.euclidean(centroids[i],centroids[j])
    Rij = (compute_s(i,x,centroids,labels,nc)+compute_s(j,x,centroids,labels,nc))/dist
    return Rij
    
    
    
def compute_R(i,x,centroids,labels,nc):
    list_R=[]
    for i in range(nc):
        for j in range(nc):
            if(i!=j):
                Ri = compute_Rij(i,j,x,centroids,labels,nc)
                list_R.append(Ri)
    return max(list_R)
    

def compute_db(x,centroids,labels,nc):
    sum1=0.0
    for i in range(nc):
        sum1+=compute_R(i,x,centroids,labels,nc)
    DB= float(sum1)/float(nc)
    return DB

db_index = compute_db(X4,centroids,labels,n)

print(db_index)
  
