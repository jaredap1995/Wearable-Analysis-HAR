"""
CNN for HAR from IMU data 

CNN1.py shape data ready for CNN model in Keras, plot data analysis
CNN2.py CNN model using synthetic data with training perf plots
CNN3.py column 1 = label in text 
        6 activities - stand, sit, walk, run, climb gate
CNN4.py updated to work following make_traing_data4.py
        15 features included up, flat and down
"""

"""
CNN for HAR from IMU data 
"""
from itertools import cycle
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam  # importing allows adjustment of learngin rate etc
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import dstack  # convet to a 3d array, like reshape[1,8,8)
from numpy import mean
import numpy as np
from numpy import std
from numpy import dstack
import os
import pandas as pd
from pandas import read_csv
import pdb  # debugger, set_trace() for beark point.
from pdb import set_trace as bp
import scipy as spy
from scipy import interp
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_circles
from sklearn.metrics import auc  # for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import time as time


"""___ Functions ___"""


def relu_clip(x):
    return K.relu(x, max_value=1000)


""" ___ Settings ___ """

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

startTime = time.time()  # used to compute processing time of this script


plot = True  # sets if plots are shown or not.
plotAllatEnd = True  # allow complete running with no interaction.
numFeatures = 3  # this is x,y,z accelleration , NOT Classes lie run, jog,sit.


fileNum = 3
combineWalkRun = 3  # only used for fileNum=3, removes slope data dataset_trail
if fileNum == 1:
    inputDataFile = "dataset_syntheticIMU"
if fileNum == 2:
    inputDataFile = ""  # NOT IN USE
# dataset_trail is sampled at 100Hz
if fileNum == 3:
    inputDataFile = "dataset_trail"
# NOTE: WISDM file has be modeifed to remove ; from end of lines and .txt to .csv
# WISDM data is sampled at 20 HZ, window size of 16 works well.
if fileNum == 4:
    inputDataFile = "datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw"


numSamples = 10000000  #
if fileNum == 4:
    numSamples = (
        1000000
    )  # 344063 is the largest number that works with WISDM dataset, dont knwo why.

verbose = 1  # define the amount of updates printed to the screen during model training
batch_size = 256  # the number of rows before back propogation takes place
epochs = 10  # the number of times the TOTAL dataset is trained against the model

expNum = 1  # 10      #number of experiments to run
strideArr = [
    256
]  # [16,16,32,32,64,64,128,128,256,256,512,512]           #samples in each window passed to NN
overlapArr = [128]  # [8,8,16,16,32,32,64,64,128,128,256,256,]
# overlap = int(stride/2) #overlap between each window, 0.5 = 50% reuse
labelRatioArr = [
    0.8
]  # [0.2,0.8]       #stride window only used if #samples have a singel label > this
ExperimentColumns = (
    "ExpNum",
    "Epoch",
    "Batch",
    "Stride",
    "Overlap",
    "LabelRatio",
    "numFeatures",
    "numLabels",
    "numRows",
    "windowAccept",
    "windowReject",
    "Accuracy",
    "PreLay",
    "PreSit",
    "PreClimb",
    "PreWalk",
    "PreRun",
)
dfExpResults = pd.DataFrame(columns=ExperimentColumns)

"""Start"""
print("------------------------------------\n\n\n STARTING CNN4.py...")

print("load data set ", inputDataFile)
# LOAD DATA
# rem in python...a()=tuple, a[]=list
# compatible with actigraph database - enables comparisons
columnNames = ["userID", "label", "timestamp", "x-axis", "y-axis", "z-axis"]
df = pd.read_csv(
    inputDataFile + ".csv", header=None, names=columnNames, nrows=numSamples
)


# drop rows with all zeros
df[df.values[:, 3:6].sum(axis=1) != 0]

print("Total number of samples before masking = ", len(df))


# ___ OPTION TO RESTRICT DATA TO CERTAIN LABELS ___
"""
# Load Labels codex - only used as comparison in a print statement below. 
dfLabelCodex = pd.read_csv (inputDataFile+str('_label_codex.csv'),header=None, sep=',')
labelCodex = dfLabelCodex.values[:,0]
"""

if fileNum == 3:  # only on the trail running dataset to tune what we analyse.

    if combineWalkRun == 1:  # remove up and down from walk and run
        print("[INFO] combineWalkRun = True, combining down/flat/up")
        df["label"] = df["label"].replace(["walk down", "walk up"], "walk")
        df["label"] = df["label"].replace(["run down", "run up"], "run")

    if combineWalkRun == 1:
        labelMask = ["lay", "sit", "climb gate", "walk", "run"]
    else:
        labelMask = [
            "lay",
            "sit",
            "climb gate",
            "walk",
            "walk up",
            "run down",
            "run",
            "run up",
        ]  # choose what features to keep
    print("Data set limited to the following labels\n", labelMask)
    for n in range(len(labelMask)):  # loop through the labelMask and apply to the mask.
        if n == 0:
            mask = df["label"] == labelMask[n]
        else:
            mask = mask | (df["label"] == labelMask[n])

    #  mask = (df['label']==labelMask[0]) | (df['label']==labelMask[1] )| (df['label']==labelMask[2] )|(df['label']==labelMask[3] ) | (df['label']==labelMask[4] )

    df = df[mask]

"""f (fileNUm ==4):
    df.values["""

# ___ MOVE FROM DF TO DATA AND LABEL
data = df.values[:, 3:6].astype("float64")  # load training features accel,x,y,zcat

# Normalise data
dataMin = min(min(data[:, 0]), min(data[:, 1]), min(data[:, 2]))
dataMax = min(max(data[:, 0]), max(data[:, 1]), max(data[:, 2]))

data[:, 0] = (data[:, 0] - dataMin) / (dataMax - dataMin)
data[:, 1] = (data[:, 1] - dataMin) / (dataMax - dataMin)
data[:, 2] = (data[:, 2] - dataMin) / (dataMax - dataMin)

numSamples = len(data)
labels = df.values[:, 1]  # load all the text based labels, like 300,000 rows..
print("number of samples = ", numSamples)
if (
    fileNum == 3
):  # if dataset_trail then use the labelMask to maintain order. e.g. sit, stand...
    # catagorical will change this to alphabetical ordering whcih is not logical
    # when viewing results.
    unq = np.array(labelMask)
else:
    cat = pd.Categorical(labels)  # find what labels are in the label array
    unq = np.unique(
        cat
    )  # get one of each of the labels <--- THIS IS WHERE ORDERING TAKES PLACE


"""if( fileNum == 4): #Actigraph
    labelMask = ['Sitting','Standing','Upstairs','Downstairs','Walking','Jogging']
    unq = np.array(labelMask)
"""


labelCodex = unq
numClasses = unq.shape[0]  # total number of classes in the dataset
print("number of classes in THIS dataset with mask", numClasses)

for i in range(numSamples):  # replace labels from text to numbers
    for k in range(len(unq)):
        if labels[i] == unq[k]:
            labels[i] = k

labels = labels.reshape(len(labels), 1)  # make an array of nx1

# ___ NORMALISE DATA ___

mu = np.mean(data, axis=0)
sigma = np.std(data, axis=0)
maxData = np.max(data, axis=0)
minData = np.min(data, axis=0)

# normalise between 0 to 1
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

# plot the data being used

plt.figure(figsize=(12, 10))
plt.plot(labels)
plt.plot(data[:, 0])
plt.show()


# run the experiment
ExpCnt = 0  # ExperimentCount
for strideCount in range(len(strideArr)):

    for labelRatioCount in range(len(labelRatioArr)):

        stride = strideArr[strideCount]
        overlap = overlapArr[
            strideCount
        ]  # <--- stride and overlap change together.....
        labelRatio = labelRatioArr[labelRatioCount]

        # ___ SEGMENT DATA FOR NN INGESTION ___
        # sement data into 3D array (count, data_of_width_"stride", number_of_features)
        # e.g. testX.shape (4323, 128, 3) is stride = 128 and 3 features, say x,y,z

        print("segment data by stride ", stride, " with overlap = ", overlap)

        X, y = list(), list()

        windowAccept = 0
        windowReject = 0
        for n in range(
            0, len(data), overlap
        ):  # run through data rows with step = stride length
            # check we are not passed the end of the data
            end_ix = n + stride
            if end_ix > len(data) - 1:
                break

            tempHistogram = np.zeros(numClasses)
            for m in range(n, end_ix):
                tempHistogram[int(labels[m])] += 1

            # total number of the maximum labels.
            labelMaxValue = tempHistogram.max()

            # BUG FIX for when there are two values equal (e.g. 128 in two places
            if labelMaxValue > int(
                stride / 2
            ):  # ONLY use labels with > 50% useage in the window

                # location where max number of activties are located

                labelMaxUsed = int(np.where(tempHistogram == tempHistogram.max())[0])

            else:
                labelValue = 0

            # NOTE!!! label is chosen as center of stride window.
            # label is middle sample seq_x, seq_y = data[n:end_ix],labels[n+np.int(stride/2)] #
            # label is max used label in window
            seq_x, seq_y = data[n:end_ix], labelMaxUsed

            # FILTER WINDOWS BASED ON LABEL CONTENT FOR EACH SAMPLE
            # find the number of most common labels.
            # tempDF = pd.DataFrame(labels[n:n+end_ix]).groupby(0).size()
            # tempMAX = tempDF.max() #the most counts

            if (
                labelMaxValue / stride >= labelRatio
            ):  # decide if enough of one label is in this window
                X.append(seq_x)
                y.append(seq_y)
                windowAccept += 1
            else:
                windowReject += 1
        print("Rejection label ration = ", labelRatio)
        print("accepted windows = ", windowAccept)
        print("rejected windws = ", windowReject)

        X = np.array(X)  # change from python list to numpy array
        y = np.array(y)
        print("shape X and y = ", X.shape, y.shape)
        print("Total samples reduced to ", X.shape[0] * overlap)

        # ___ BALANCE OF ACTIVITY CLASSES ___
        # summarise the number of each class
        # make dataframe of label array
        print("Class analysis of dataset")
        df = pd.DataFrame(labels)
        # group the labels
        counts = df.groupby(
            0
        ).size()  # groupby col 1 and return the size of each group.
        # 1labelCount = len(numClasses)#np.array([1,3])

        # convert to numpy array
        # another cool way if len(np.unique(labels)) ids

        # NOTE there may not be an example of each label in the dataset.
        counts = np.array(counts)
        numCodex = len(labelCodex)
        numClasses = len(counts)
        print(" Number of classes IN THIS DATASET = ", numClasses)
        print(" Number of TOTAL CLASSES AVAILABLE = ", numCodex)
        # sumarise
        for n in range(numClasses):  # loop through the number of groups
            percent = counts[n] / len(df) * 100  # calculate the percent of all data
            print(
                "Class=%d, %s total=%d, percentage=%0.2f"
                % (n + 1, labelCodex[n], counts[n], percent)
            )

        # ___ MAKE LABELS CATAGORICAL
        print("make labels catagorical")
        y = to_categorical(y)

        # note: trian_test_split allocates fairly each catagory between test and train.
        print("train test split")
        trainX, testX, trainy, testy = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        # ___DEFINE MODEL
        def get_model(trainX, trainy):

            # fit and evaluate a model

            n_timesteps, n_features, n_outputs = (
                trainX.shape[1],
                trainX.shape[2],
                trainy.shape[1],
            )

            model = Sequential()
            model.add(BatchNormalization())  # added 190714s
            leakyrelu = LeakyReLU(alpha=0.1)
            model.add(
                Conv1D(
                    filters=64,
                    kernel_size=3,
                    activation=relu_clip,
                    input_shape=(n_timesteps, n_features),
                )
            )
            model.add(Conv1D(filters=64, kernel_size=3, activation=relu_clip))
            model.add(Dropout(0.5))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(100, activation="relu"))  # replaced activation='relu'
            model.add(Dense(1))
            model.compile(optimiszer="adam", loss="mse")
            model.add(Dense(n_outputs, activation="softmax"))  # classifier
            keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(
                loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
            )

            return model

        print("define model")
        model = get_model(trainX, trainy)
        # fit network

        print("fitting model... ")
        history = model.fit(
            trainX,
            trainy,
            epochs=epochs,
            validation_data=(testX, testy),
            batch_size=batch_size,
            verbose=verbose,
        )

        # evaluate the model
        print("evaluate mode")
        _, train_acc = model.evaluate(trainX, trainy, verbose=0)
        _, test_acc = model.evaluate(testX, testy, verbose=0)

        # ___ PERFORMANCE SUMMARY ___

        print("Epochs, Batchsize = ", epochs, batch_size)

        print("[INFO] model = ", model)
        for layer in model.layers:
            print("         ", layer.name)

        # predict probabilities for test set
        yhat_probs = model.predict(testX, verbose=0)

        # predict crisp classes for test set
        yhat_classes = model.predict_classes(testX, verbose=0)
        # reduce to 1d array - for sklearn API
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:]

        # reverse one hot encoding for testy - SKLern API doesnt like one hot encoded
        testy_1d = np.zeros(len(testy))
        for i in range(len(testy)):
            for k in range(numClasses):
                if int(testy[i, k]) == 1:
                    testy_1d[i] = k

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(testy_1d, yhat_classes)
        print("Accuracy:  ", accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(
            testy_1d, yhat_classes, average=None
        )  # average=None for multiclass. default is 'average' for binary classification.
        print("Precision: ", precision)
        # recall: tp / (tp + fn)
        recall = recall_score(testy_1d, yhat_classes, average=None)
        print("Recall:    ", recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(testy_1d, yhat_classes, average=None)
        print("F1 score:  ", f1)

        # kappa
        kappa = cohen_kappa_score(testy_1d, yhat_classes)
        print("Cohens kappa: ", kappa)
        # ROC AUC

        # confusion matrix
        matrix = confusion_matrix(testy_1d, yhat_classes)
        print(matrix)

        y_score = model.predict_proba(testX)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(numClasses):
            fpr[i], tpr[i], _ = roc_curve(testy[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(testy.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numClasses)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(numClasses):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= numClasses

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        #'ExpNum','Epoch','Batch','Stride','Overlap','LabelRatio','Accuracy','numFeatures','numLabels','numRows')

        dfExpResults.loc[ExpCnt] = [
            ExpCnt,
            epochs,
            batch_size,
            stride,
            overlap,
            labelRatio,
            numFeatures,
            numClasses,
            numSamples,
            windowAccept,
            windowReject,
            accuracy,
            precision[0],
            precision[1],
            precision[2],
            precision[3],
            precision[4],
        ]

        print("results so far\n", dfExpResults)

        ExpCnt += 1


# _____ END EXPERIMENTS _____
# ___ PLOT BALANCE OF ACTIVITY ___
if plot == 1:
    df = pd.DataFrame(labels)
    counts = df.groupby(0).size()
    plt.figure()
    plt.title("Distribution per activity")

    plt.bar(counts.index, counts)
    x = range(len(labelCodex))
    plt.xticks(x, labelCodex, rotation=70)
    # plt.xticks(range(numClasses))#len(counts)))
    if plotAllatEnd == False:
        plt.show()

# ___ PLOT EACH AXIS OVER TIME ___ DATA AND LABELS
if plot == 1:
    plt.figure()
    plt.title("Time Series Plot of features with Labels")
    # determine the total number of plots
    n, offset = X.shape[2] + 1, 0  # shape[1]=num of rows, shape[2]=num of columns
    # plot data
    for i in range(numFeatures):
        plt.subplot(n, 1, offset + 1)
        plt.plot(data[:, i])
        plt.title(
            "  total acc " + str(i), y=0, loc="left", size=7
        )  # turn off ticks to remove clutter
        plt.xticks([0, len(data)], rotation=70)
        offset += 1
    # plot activities
    plt.subplot(n, 1, n)
    plt.plot(labels[:, 0])
    plt.title(" label", y=0, loc="left", size=7)

    plt.xticks([0, len(data)])
    if plotAllatEnd == False:
        plt.show()

# ___ PLOT HISTOGRAM OF LABEL ___
if plot == 100:  # i.e. NEVER FOR NOW
    plt.figure()
    plt.title("Histogram of Values by Axis (a.k.a feature)")

    for i in range(numFeatures):
        plt.subplot(numFeatures, 1, i + 1)
        plt.xlim(-1, 1)
        plt.hist(
            np.isfinite(data[:, i]), bins=100
        )  # isfinite removes Nan and infinite numbers.
        if i == 0:
            plt.title("  x", y=0, loc="left", size=7)
        if i == 1:
            plt.title("  y", y=0, loc="left", size=7)
        if i == 2:
            plt.title("  z", y=0, loc="left", size=7)
        # plt.yticks([])
        plt.xticks([-1, 0, 1])
    if plotAllatEnd == False:
        plt.show()

# ___ PLOT OF ACCELERATION DATA BY ACTIVITY ___
if plot == 10:  # THIS DOESNT WORK WITH MISSING DATA IN EACH CLASS
    ids = np.unique(labels)

    a = {a: X[y[:, 0] == a, :, :] for a in ids}  # remove empty lines????

    plt.figure()
    plt.title("Histogram of Values by Activity")
    for k in range(len(ids)):
        act_id = ids[k]
        for i in range(3):
            ax = plt.subplot(len(ids), 1, k + 1)
            ax.set_xlim(0, 1)  # <--- MAYBE -1 to +1 sometimes
            plt.hist(a[k][0][:, i], bins=100)
            if i == 0:
                plt.title(labelCodex[k], y=0, loc="left", size=7)
    if plotAllatEnd == False:
        plt.show()

# ___ PLOT DURATION OF ACTIVITY ___
print("plot duration of each activty - to do - will be a box plot...")
if plot == True:
    # plot loss during training
    print("plot training Loss and Accuracy")
    plt.figure()
    plt.subplot(211)
    plt.title("Loss")
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], label="train")
    plt.plot(history.history["val_acc"], label="test")
    plt.legend()
    if plotAllatEnd == False:
        plt.show()
    # Plot all ROC curves
    lw = 1
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(numClasses), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Multi class")
    plt.legend(loc="lower right")
    plt.show()


print("[INFO] save experiment results to CNN4_results.csv")
dfExpResults.to_csv("CNN4_results.csv", index=False, sep=",", encoding="utf-8")

print("[INFO] save model to model.h5")
model.save("model.h5")

print("[INFO] save groundTruth to groundTruth.npy")
np.save("groundTruth.npy", testy)

print("[INFO] save test data to testData.npy")
np.save("testData.npy", testX)

# --- END OF PROGRAM PROCESSING
stopTime = time.time()  # invalid if after plt.show()
processTime = (int((stopTime - startTime) * 1000)) / 1000
print("\nprocess time = ", processTime, " s")

print("\ncomplete")
