# 5-class Hardness Classification
## 5-class hardness classification by a tactile sensing system mounted on a Baxter robot. 

The "Dataset" folder contains the cube and cylinder data divided per class for 40 and 80 samples extracted from the grasp peaks. For each kind of objects we collected five classes, i.e. five hardness levels, having 170 data each. The classes 0, 1, and 2 of the cylinders correspond to the classes 0,1, and 2 of the cubes, respectively. 

The "Python Scripts" folder contains the python scripts for generating the training and test splits from the Dataset folder and for training and testing three ML algorithms. In particular:

1. Extract_Train_Test_Splits.py lets to create the training and testing folders. It can create the splits only for the cube objects or a merged dataset consisting of all the cubes and the cylinder data that match with the cubes hardness levels.
2. ElaborationClass.py contains the three ML algorithms' implementations: a fully connected neural network, a kernel support machine, and a 1D convolutional neural network.
3. Main.py is used to train and test the algorithms based on the input dataset path that have to be specified. 
