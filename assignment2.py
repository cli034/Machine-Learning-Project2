import numpy as np
import matplotlib.pyplot as plt

#load in data
#missing values "?" is replaced with -1 for now
#force output to be integer
def loadData(fileName):
    np.set_printoptions(threshold=np.inf) #so it prints the whole thing
    breastCancerData = np.genfromtxt(fileName, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10), dtype=np.int)

    for i in range(0,np.size(breastCancerData[:,0])):
        for j in range(0,np.size(breastCancerData[0,:])):
            if (breastCancerData[i,j] == -1):
                breastCancerData[i,j] = -1 #whatever to replace missing data

    #print breastCancerData
    return breastCancerData

# Lp norm distance formula
# From assignment 1
def distance(x,y,p):
    array1 = x
    array2 = y
    temp = 0
    total = 0
    distance = 0

    for i in range(len(array1)):
        temp = abs(array1[i] - array2[i]) ** p
        total = total + temp
    
    distance = total ** (float(1)/p)
    return distance

#Question 1
#for each test compare to all of the train
#k is the number of nearest neighbors you select
#p is the parameter for Lp norm distance formula, p = 1 or p = 2
def knn_classifier(x_test, x_train, y_train, k, p):
    y_pred = []
    
    for i in range(len(x_test)):
        dist_array = dict()
        benignCnt = 0
        malignantCnt = 0
        for j in range(len(x_train)):
            dist_array[j] = distance(x_test[i], x_train[j], p)
        #sort the dictionary, returns a list of keys assocaited to the item
        sortedIndex = sorted(dist_array, key=dist_array.get)
        # returns the index of the item for number of nearest neighbors we selected
        #sortedIndex[:k]

        #this loop finds the majority of the label
        for g in sortedIndex[:k]:
            #print y_train[g]
            if (y_train[g] == 2):
                benignCnt = benignCnt + 1
            elif (y_train[g] == 4):
                malignantCnt = malignantCnt + 1
        if (benignCnt > malignantCnt):
            y_pred.append(2)
        else:
            y_pred.append(4)
    return y_pred

#Question 2
#shuffling data
def shuffleData(dataSet):
    np.random.shuffle(dataSet)
    return breastCancerData

#compute the perfomance, accuracy, sensitivity, and specificity
def performance(actual, predict):
    #malignant = positive (means breast cancer, represented by value 4)
    #benign = negative (means no breast cancer, represented by value 2)
    count = 0
    truePositive = 0
    falsePositive = 0
    falseNegative = 0
    trueNegative = 0

    for i in range(len(actual)):
        if (actual[i] == predict[i]):
            count = count + 1
        if ((actual[i] == 4) and (predict[i] == 4)):
            truePositive = truePositive + 1
        if ((predict[i] == 4) and (actual[i] == 2)):
            falsePositive = falsePositive + 1
        if ((predict[i] == 2) and (actual[i] == 4)):
            falseNegative = falseNegative + 1
        if ((predict[i] == 2) and (actual[i] == 2)):
            trueNegative = trueNegative + 1
        
    
    accuracy =  count / float(len(actual)) * 100
    sensitivity = truePositive / float((truePositive + falseNegative))
    specificity = trueNegative / float((trueNegative + falsePositive))

    print "Accuracy: " + str(count) + "/" + str(len(actual)) + " = " + str(accuracy) + "%"
    print "Sensitivity: " + str(sensitivity)
    print "Specificity: " + str(specificity)

    return (accuracy, sensitivity, specificity)


#Question 2
#k value is how many times we fold
def k_fold_cross_validation(dataSet, neighbors, k, p_value):
    data = shuffleData(dataSet)
    sections = data.shape[0] / k
    starting = 0

    accuaryList = []
    sensitivityList = []
    specificityList = []

    array_fold = [] #this is just an array for the x-axis of the graph

    for i in range(1, k + 1):
        accuracyTemp = 0
        sensitivityTemp = 0
        specificityTemp = 0

        testingSet = data[starting:sections * i, :9]
        #stores the acutal label for the testing set and compare to predicted
        actualLabels = data[starting:sections * i, 9:]
        #cut out the testing set from the data set the the rest are training dats
        #found the slice funtion on stackoverflow
        trainingSet = np.delete(data, slice(starting, (sections * i)), axis=0)[:, :9]
        #stores the label for the training data
        trainingSetLabels = np.delete(data, slice(starting, (sections * i)), axis=0)[:, 9:]
        y_pred = knn_classifier(testingSet, trainingSet, trainingSetLabels, neighbors, p_value)
        starting = sections * i
        print y_pred

        (accuracyTemp, sensitivityTemp, specificityTemp) = performance(actualLabels, y_pred)
        accuaryList.append(accuracyTemp)
        sensitivityList.append(sensitivityTemp)
        specificityList.append(specificityTemp)
        array_fold.append(i)
    
    accuracyArray = np.array(accuaryList)
    sensitivityArray = np.array(sensitivityList)
    specificityArray = np.array(specificityList)

    print "\n"
    print "\nAccuracy Mean: " + str(np.mean(accuracyArray))
    print "Accuracy Standard Deviation: " + str(np.std(accuracyArray))
    print "\nSensitivity Mean: " + str(np.mean(sensitivityArray))
    print "Sensitivity Standard Deviation: " + str(np.std(sensitivityArray))
    print "\nSpecificity Mean: " + str(np.mean(specificityArray))
    print "Specificity Standard Deviation: " + str(np.std(specificityArray))

    #plot x (number of nearest neighbor) vs y (performance)
    plt.plot(array_fold, accuracyArray)
    #plt.errorbar(accuracyArray, np.std(accuracyArray))
    plt.title("Accuracy, P = " + str(p_value))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of nearest neighbors: " + str(neighbors))
    plt.errorbar(array_fold, accuracyArray, yerr=np.std(accuracyArray), fmt='.')
    plt.show()

    plt.plot(array_fold, sensitivityArray)
    plt.title("Sensitivity, P = " + str(p_value))
    plt.ylabel("Sensitivity")
    plt.xlabel("Number of nearest neighbors: " + str(neighbors))
    plt.errorbar(array_fold, sensitivityArray, yerr=np.std(sensitivityArray), fmt='.')
    plt.show()

    plt.plot(array_fold, specificityArray)
    plt.title("Specificity, P = " + str(p_value))
    plt.ylabel("Specificity")
    plt.xlabel("Number of nearest neighbors: " + str(neighbors))
    plt.errorbar(array_fold, specificityArray, yerr=np.std(specificityArray), fmt='.')
    plt.show()
    
    

fileName = 'breast-cancer-wisconsin.data.txt'
breastCancerData = loadData(fileName)
#the professor wants us to split the data 80/20 (first 80% of the data is the training set and the rest 20% is the testing set)
trainingSetSize = int(breastCancerData.shape[0] * 0.8)
#testingSetSize = int(breastCancerData.shape[0] - trainingSetSize)
x_train = breastCancerData[:trainingSetSize, :9]
x_test = breastCancerData[trainingSetSize:, :9]
y_train = breastCancerData[:trainingSetSize, 9:]

option = raw_input("Press 1 to run knn_classifier, press 2 to run k-fold cross validation: ")
if (option == "1"):
    k_value = input("Enter k value for the classifier: ")
    p_value = input("Enter p value for the classifier: ")
    print knn_classifier(x_test, x_train, y_train, k_value, p_value)
elif (option == "2"):
    neighbor_nums = input("Enter number of nearest neighbors you are going to use: ")
    userinput = raw_input("Enter p value for knn_classifier: ")
    if (userinput == "1"):
        k_fold_cross_validation(breastCancerData, neighbor_nums, 10, 1)
    elif (userinput == "2"):
        k_fold_cross_validation(breastCancerData, neighbor_nums, 10, 2)
    