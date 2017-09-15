
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import math
import random
import time
import heapq

def distance(data,centroids,p1):
    return [euclidean(data[j,:] , p1) for j in centroids]

def getCentroids(data,k,n):
    #getting centroids using kmeans++
    print("\nGetting Centroids using Kmeans++....")
    points = np.arange(n)
    centroids = list(np.random.choice(points , 1))
    points = np.setdiff1d(points,centroids)

    run = 1
    while (run < k):
        mega = [np.square(np.amin(distance(data,centroids,data[i,:]))) for i in range(len(points))]
        new = np.random.choice(points,1,replace=False,p= (mega/np.sum(mega)))
        run+=1
        centroids.append(new[0])
        points = np.setdiff1d(points,new)

    return centroids

def getCheat(data,k,n):
    #getting the mean and using that for the cheat function
    true_labels = readData("number_labels.txt",1)
    clusters_unique = np.unique(true_labels)
    centroids = []

    for label in clusters_unique:
        indicies = np.where(true_labels == label)
        act = []
        for i in indicies:
            act.append(data[i,:])
        cent = np.mean(act[0],axis=0)
        centroids.append(cent)

    return np.array(centroids)



def move_centroids(points, closest, centroids_data):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids_data.shape[0])])

def euclidean(point1,point2):
    return np.linalg.norm(point1 - point2)
    # return np.sqrt(np.sum((point1-point2)**2)) #faster for now

def SSE(data,centroids_data,labels):
    distances = [euclidean(data[k], centroids_data[labels[k]]) for k in range(len(data))]
    return(np.sum(np.square(distances)))



def assignLabels(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:,np.newaxis])**2).sum(axis=2))
    closest = np.argmin(distances, axis=0)

    #making sure there is no empty clusters
    cluster_sizes =  (np.array([len(points[closest==k]) for k in range(centroids.shape[0])]))
    zero_clusters = np.where(cluster_sizes == 0)[0]
    zero_len = len(zero_clusters)

    #Check for empty clusters and do something if found
    while (zero_len > 0):

        print ("Fixing empty clusters...")
        act_distances = np.min(distances,axis=0)
        to_replace = (-act_distances).argsort()[:zero_len] 

        for i in range(zero_len):
            centroids[zero_clusters[i],:] = points[to_replace[i],:]
        
        distances = np.sqrt(((points - centroids[:,np.newaxis])**2).sum(axis=2))
        closest = np.argmin(distances, axis=0)

        #making sure there is no empty clusters
        cluster_sizes =  (np.array([len(points[closest==k]) for k in range(centroids.shape[0])]))
        zero_clusters = np.where(cluster_sizes == 0)[0]
        zero_len = len(zero_clusters)


    return closest

def display_image(c,i):
    #displays a list of 784 gray scale values as an image
    data = np.array(c)
    data = np.reshape(data,(-1,28))
    plt.imshow(data)
    plt.savefig(("IM"+i+".png"))


def readData(name,which):
    #read data
    data = []
    f= open(name)

    fil = f.readline()
    if (which  == 0):
        while fil:
            data.append(np.array([int(i) for i in fil.split(',')]))
            fil = f.readline()
    else:
         while fil:
            data.append(int(fil))
            fil = f.readline()       

    f.close()
    return np.array(data)


def main():
    #main
    print("LOADING DATA ...")
    start_time = time.time()
    data = readData("number_data.txt",0)
    n = np.shape(data)[0]
    print ("Data Size : ",n)
    print("--- %s seconds to LOAD DATA---\n" % (time.time() - start_time))


    print("GETTING INITIAL CENTROIDS ...")
    start_time = time.time()
    #check argments
    if len(sys.argv) < 3:
        print("Not enough args")
        exit()
    else:
        k = int(sys.argv[1])
        mode =  sys.argv[2]
        #check if they want random or
        if (mode == "random"):
            centroids = random.sample(range(0,n), k)
            centroids_data = []

            for c in centroids:
                centroids_data.append(data[c,:])

            centroids_data = np.array(centroids_data)

        elif (mode == "cheat"):
            centroids_data = getCheat(data,k,n)

        else:
            centroids = getCentroids(data, k,n)
            centroids_data = []

            for c in centroids:
                centroids_data.append(data[c,:])

            centroids_data = np.array(centroids_data)


    # print("\nInitial Centroids : ", centroids)
    print("\n--- %s seconds to GET CENTROIDS---" % (time.time() - start_time))

    print("\nRUNNING K-MEANS ...")
    
    start_time = time.time()
    iterations = 0

    stop = True
    labels = assignLabels(data, centroids_data)

    while (stop or iterations == 0):
        iterations += 1
        oldLabels = labels

        centroids_data = move_centroids(data, labels, centroids_data)
        labels = assignLabels(data, centroids_data)

        sse = (SSE(data,list(centroids_data),list(labels)))
        
        print("On iteration ", iterations, ", the SSE was : ",sse)

        if (np.array_equal(oldLabels,labels)):
            stop = False

    print("\n--- %s seconds to RUN K-MEANS---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
