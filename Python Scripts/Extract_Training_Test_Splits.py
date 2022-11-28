import glob
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

samples = 80 # number of samples of grasp peaks. It can be either 40 or 80
assert samples == 80 or samples == 40, "THE NUMBER OF CHOSEN SAMPLES IS WRONG"
if samples == 80:
    samplesStr = "80_Samples"
else:
    samplesStr = "40_Samples"
datasetCube = os.path.join("../Dataset/Cube", samplesStr) # dataset path from where splits the cube data
assert os.path.exists(datasetCube), "THE CUBE DATASET DOES NOT EXIST"

percTestCube = 50/170 # percentage of cubes used for testing
percTestCylinder = 130/170 # percentage of cylinders used for testing. It is used only for the merged dataset
numberOfCubesToMove = int(170*(1-percTestCylinder)) # number of cubes to move from the training folder to the testing folder
classesCylindersToTake = [0, 1, 2] # cylinders hardness classes to be taken for training and test. These three classes match with the ones of the cubes
seed = 666  # seed for the random extraction of the data for training and test
mergedDataset = True # False if one wants to create only the cube dataset. True to create the merged dataset
if mergedDataset:
    datasetCylinder = os.path.join("../Dataset/Cylinder", samplesStr)  # dataset path from where splits the cylinder data
    rootDataset = "../Dataset/Merged" # root path for the merged split dataset
else:
    rootDataset = "../Dataset/CubesOnly" # root path for the cube split dataset

# create the training and test folders
trainingPath = os.path.join(rootDataset, samplesStr, "train")
testingPath = os.path.join(rootDataset, samplesStr, "test")
if os.path.exists(trainingPath):
    shutil.rmtree(trainingPath)
    os.makedirs(trainingPath)
else:
    os.makedirs(trainingPath)
if os.path.exists(testingPath):
    shutil.rmtree(testingPath)
    os.makedirs(testingPath)
else:
    os.makedirs(testingPath)
# ###############################################################


def split_cubes_for_Merged(Xtrain, ytrain, Xtest, ytest, cylinderClasses, numToMove):
    """
    function used for moving some cubes from the training folder to the test folder when the hardness classes between cubes and cylinders match
    :param list Xtrain: training data for the cubes
    :param ndarray ytrain: training labels corresponding to the training data
    :param list Xtest: testing data for the cubes
    :param ndarray ytest: testing labels corresponding to the testing data
    :param list cylinderClasses: the cylinder classes that match with the cube classes
    :param int numToMove: number of cubes to move from the training folder to the test folder
    :return: the new train and test splits for the cubes
    """
    indexesToMove = [] # list to collect the indexes of the data to move from the training to the test folder
    indexesToTake = [] # list to collect the indexes of the data to take for testing
    classes = np.unique(ytrain) # extract automatically the classes of the cubes
    # for loop to decide which cube data have to be moved from the train to the test folder for each class of the cylinders
    for c in cylinderClasses:
        tmp = np.where(ytrain == c)[0]
        np.random.shuffle(tmp)
        indexesToMove.extend(tmp[:numToMove])
        indexesToTake.extend(tmp[numToMove:])
    # take all the cube data that are a different classes with respect to the cylinders
    for c in classes:
        if c not in cylinderClasses:
            tmp = np.where(ytrain==c)[0]
            indexesToTake.extend(tmp)
    ytest = list(ytest)
    # move the training data to test
    for i in indexesToMove:
        Xtest.append(Xtrain[i])
        ytest.append(ytrain[i])
    # change the objects from list to numpy array
    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytest = np.asarray(ytest)
    indexesToTake = np.asarray(indexesToTake)
    return Xtrain[indexesToTake], Xtest, ytrain[indexesToTake], ytest


if mergedDataset:
    globalPathCubes = glob.glob(datasetCube + "/*/*") # retrieve all the cube data
    globalPathCylinders = glob.glob(datasetCylinder + "/*/*") # retrieve all the cylinders data
    globalPathCubes.sort() # sort the cube data
    globalPathCylinders.sort() # sort the cylinder data
    assert len(globalPathCubes) > 0, "THE CUBE DATASET IS EMPTY" # check if the cubes path is not empty
    assert len(globalPathCubes) > 0, "THE CYLINDER DATASET IS EMPTY" # check if the cylinder path is not empty
    yCubes = np.asarray([int(i.split('/')[-2]) for i in globalPathCubes]) # retrieve the cube labels
    yCylinders = np.asarray([int(i.split('/')[-2]) for i in globalPathCubes]) # retrieve the cylinder labels
    # split the cubes in training and test maintaining the balance between the classes through the stratify parameter
    trainFilesCubes, testFilesCubes, yTrainCubes, yTestCubes = train_test_split(globalPathCubes, yCubes, stratify=yCubes, test_size=percTestCube,
                                                                                random_state=seed)
    # split the cylinders in training and test maintaining the balance between the classes through the stratify parameter
    trainFilesCylinders, testFilesCylinders, yTrainCylinders, yTestCylinders = train_test_split(globalPathCylinders, yCylinders,
                                                                                stratify=yCylinders, test_size=percTestCylinder,
                                                                                random_state=seed)
    # move as many cubes from the train to the test folder as cylinders will be used for training
    trainFilesCubes, testFilesCubes, yTrainCubes, yTestCubes = split_cubes_for_Merged(trainFilesCubes, yTrainCubes, testFilesCubes, yTestCubes, classesCylindersToTake, numberOfCubesToMove)
    classes = np.unique(yCubes)
    # create the folders for each class
    for c in classes:
        os.mkdir(os.path.join(trainingPath, str(c)))
        os.mkdir(os.path.join(testingPath, str(c)))
    # move the data from the original dataset to the training and testing split for both the objects
    for i, tf in enumerate(trainFilesCubes):
        cl = yTrainCubes[i]
        name = tf.split('/')[-1]
        shutil.copyfile(tf, os.path.join(trainingPath, str(cl), name))
    for i, tf in enumerate(testFilesCubes):
        cl = yTestCubes[i]
        name = tf.split('/')[-1]
        shutil.copyfile(tf, os.path.join(testingPath, str(cl), name))
    for i, tf in enumerate(trainFilesCylinders):
        cl = yTrainCylinders[i]
        if cl not in classesCylindersToTake:
            continue
        name = tf.split('/')[-1]
        shutil.copyfile(tf, os.path.join(trainingPath, str(cl), name))
    for i, tf in enumerate(testFilesCylinders):
        cl = yTestCylinders[i]
        if cl not in classesCylindersToTake:
            continue
        name = tf.split('/')[-1]
        shutil.copyfile(tf, os.path.join(testingPath, str(cl), name))
else:
    globalPath = glob.glob(datasetCube+"/*/*")
    globalPath.sort()
    assert len(globalPath) > 0, "THE DATASET IS EMPTY"
    y = np.asarray([int(i.split('/')[-2]) for i in globalPath])
    trainFiles, testFiles, yTrain, yTest = train_test_split(globalPath, y, stratify=y, test_size=percTestCube, random_state=seed)
    classes = np.unique(y)
    for c in classes:
        os.mkdir(os.path.join(trainingPath, str(c)))
        os.mkdir(os.path.join(testingPath, str(c)))
    for i, tf in enumerate(trainFiles):
        cl = yTrain[i]
        name = tf.split('/')[-1]
        shutil.copyfile(tf, os.path.join(trainingPath, str(cl), name))
    for i, tf in enumerate(testFiles):
        cl = yTest[i]
        name = tf.split('/')[-1]
        shutil.copyfile(tf, os.path.join(testingPath, str(cl), name))
print("DATASET CREATED")