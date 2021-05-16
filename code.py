import sklearn.datasets as sk
from sklearn import svm
import numpy as np
import sys
import math as mth
from numpy.linalg.linalg import norm, transpose
import matplotlib.pyplot as plt
import statistics as stats

def Online_algo(x,y):
    feature_number = x.shape[0]
    ## number of features equals to the number of input rows
    data_size = x.shape[1]
    ## number of input data equals to the number of input matrix columns
    x = transpose(x)

    w_omega = np.random.uniform(low=-1, high=1, size=(feature_number,))
    ## generate random numbers of omega from range [-1 to 1]
    E =sys.float_info.epsilon
    ## initialize epsilon which will be compared to the minimum error
    Delta = np.ones(feature_number) # Just to enter the while loop
    list_of_omega_changes = [] # is a list of all the omegas undergoing change
    list_of_Delta_changes = [] # is a list of all deltas that will change the omega
    number_of_epochs1 =0 # iterator for the number of times the delta was being updated
    while (norm(Delta,1) > E):
        Delta = np.zeros(feature_number) # intialize the delta by zeros
        
        for i in range(data_size):
            predicted_value = np.dot(w_omega,x[i]) ## get the learning function
            signn= y[i] * predicted_value ## different sign detector
            if signn <= 0 :
                ## as long as the signs of the prediction and the true are different enter this loop and update yhr omega
                Delta = (Delta - (y[i]*x[i])) 
                Delta = Delta / data_size
                list_of_Delta_changes.append(Delta)
                w_omega = w_omega - Delta
                list_of_omega_changes.append(w_omega)
                ## make this calculations to minimize delta and use it to minimize omega
        number_of_epochs1 +=1
    return w_omega , number_of_epochs1, len(list_of_omega_changes),list_of_Delta_changes

def Batch_algo(x,y):
    ## same algorithm as the online_algo but only different that omega is updated once after making the final update for the delta
    feature_number = x.shape[0]
    data_size = x.shape[1]
    x = transpose(x)

    w_omega = np.random.uniform(low=-1, high=1, size=(feature_number,))
    E =sys.float_info.epsilon
    Delta = np.ones(feature_number) # Just to enter the while loop
    list_of_omega_changes = []
    list_of_Delta_changes = []

    while (norm(Delta) > E):
        Delta = np.zeros(feature_number)
        for i in range(data_size):
            predicted_value = np.dot(w_omega,x[i])
            signn= y[i] * predicted_value
            if signn <= 0 :
                Delta = (Delta - (y[i]*x[i]))    

    ### Only difference is that here the delta and the omega are updated after reaching maximum delta optimization               
        Delta = Delta/data_size
        w_omega = w_omega - Delta

        list_of_Delta_changes.append(Delta)
        list_of_omega_changes.append(w_omega)

        number_of_epochs = len(list_of_omega_changes)
    return w_omega ,number_of_epochs ,len(list_of_omega_changes), list_of_Delta_changes

def accuracy(testing,actual):
    ## function that calculates the ratio of true testing to the whole testing data
    count = 0 
    eff = 0
    for test in range(len(testing)):
        ## loops at the testing data and increment the count if the testing is equal to the actual Y
        if np.sign(testing[test]*actual[test]) == 1.0:
            count += 1
    return (count/len(actual))*100

def draw_class(w,x,y):
    ## takes the omega , feature , output and plots the classification line witch the data points
    m = -w[0] / w[1]
    y_y = m*x
    
    f2=plt.figure()
    plt.plot(x, y_y, 'm')
    f2.suptitle("Classifier plot")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.scatter(x[:,0] ,x[:,1] , marker='o', c=y , s=25,edgecolors='k')
def make_test(x,y,s,type):
    ## Main function for testing accuracy and plotting the deltas of the given data
    ## s is a string that indicates the type of the learning algorithm
    ## type is an indicator if the data is comming from the user
    mask_for_y = y ==0
    y[mask_for_y] = -1
    ## if data of the output Y is classified as 1,0 this makes it 1,-1
    if type == "u":
        x=np.transpose(x)       
    train_count=int( 0.75 * len(x))
    ## get the 75% length needed for the training of data
    test_count = int(len(x)-train_count)
    ## get the 25% length needed for testing

    testing=[]
    # initialize an array of the data the was not given for training and that will be tested
    if s =="online":
        w1,n1,n2,n3=Online_algo(np.transpose(x[:train_count]),np.transpose(y[:train_count]))
    else:
        w1,n1,n2,n3=Batch_algo(np.transpose(x[:train_count]),np.transpose(y[:train_count]))
    ## plot the deltas 
    means_n3=[]
    for i in range(len(n3)):
        means_n3.append(np.mean(n3[i]))
    x_r=range(len(n3))
    fig = plt.figure()

    f1=plt.figure()
    plt.plot(x_r,means_n3)
    f1.suptitle("Delta plot")
    plt.ylabel("Delta mean value")
    plt.xlabel("Iteration")
    if type != "u":
        draw_class(w1,x,y)
    plt.show()
    ## testing the data by getting its predicted Y of untrained x and appending it to the testing array
    for i in range(train_count,len(x)):
        testing.append(mth.copysign(1,np.dot(w1,x[i])))
    ## call the accuracy function and printing the accuracy    
    print("Accuracy of test:",accuracy(testing,y[train_count:len(x)]))   
    ## Print the number of epoches
    print("Number of epoches for this test:",n1) 



### DATA
### prob 1
a =np.array([[50, 55, 70, 80, 130, 150, 155, 160], [1, 1, 1, 1, 1, 1, 1, 1] ])
b = np.array([1, 1, 1, 1, -1, -1,-1,-1])

## problem 4
c= np.array([[0,255,0,0,255,0,255,255],[0,0,255,0,255,255,0,255],[0,0,0,255,0,255,255,255],[1,1,1,1,1,1,1,1] ])
d= np.array([+1, +1, +1, -1, +1, -1, -1, +1])

          ###########################################################################
          ###########################################################################
          ############################################################################
          ####### TESTING THE FUNCTIONS ################
          ##### UNCOMMENT THE DESIRED TEST #######################
## problem 1 ##
#make_test(a,b,"online","u")

#make_test(a,b,"batch","u")

## problem 4 ##
#make_test(c,d,"online","u")

#make_test(c,d,"batch","u")


  
## make_classification 
## data
x,y = sk.make_classification(25,n_features= 2,n_redundant = 0,n_informative = 1,n_clusters_per_class=1)
## processing

#make_test(x,y,"online","n")

make_test(x,y,"batch","n")





