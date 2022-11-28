import os
import glob
import pickle
import enlighten
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Load the dataset. The function takes as input the dataset path
def load_dataset(datasetPath):
    assert os.path.exists(datasetPath), "DATASET PATH DOES NOT EXIST"
    globalFiles = glob.glob(datasetPath + '/*/*') # the dataset must be divided in folders representing the classes
    assert len(globalFiles) > 0, "DATASET PATH DOES NOT CONTAIN ANY DATA"
    globalFiles.sort()
    X = []  # empty list of the dataset
    y = [int(i.split('/')[-2]) for i in globalFiles] # take labels from the names of the folder
    for i in globalFiles:
        assert i.endswith(".csv"), "THE FILE HAS NOT .csv EXTENSION"
        tmp = pd.read_csv(i, header=None).values  # read the data
        X.append(tmp) # appending the datum
    X = np.asarray(X)  # transform the list of data into a numpy array
    assert len(X[0].shape) == 2, "THE DATA DIMENSION MUST BE 2 (number of samples x channels)"
    y = np.asarray(y)  # transform the list of labels into a numpy array
    print("DATASET LOADED")
    return X, y


# Extract average and std as features
def extract_feature(datasetPath):
    X_, y_ = load_dataset(datasetPath)
    X = []
    # for each datum extract the average and the standard deviation along the channels. The output will have dimension (number of data x channels*2)
    for x in X_:
        x1 = np.mean(x, axis=0)
        x2 = np.std(x, axis=0)
        X.append(np.hstack((x1, x2)))
    print("FEATURES EXTRACTED")
    return X, y_


# Plot the confusion matrix. The function takes as input the number of classes,
# the true labels, and the predicted labels
def VisualizeConfMatrix(n_classes, true_labels, y_pred):
    labels = np.asarray([i for i in range(n_classes)])
    cm = confusion_matrix(true_labels, y_pred, labels=labels)
    classesNames = [str(i) for i in range(n_classes)]
    cm_df = pd.DataFrame(cm, index=classesNames, columns=classesNames)
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

    sns.set(font_scale=2.5)
    s = np.sum(cm)
    sns.heatmap(cm_df / s, annot=cm, cmap="Greys", annot_kws={"size": 36}, cbar=False, linewidths=1,
                linecolor='black',
                fmt='g', square=True)
    plt.ylabel('True labels', fontsize=28, fontweight="bold")
    plt.xlabel('Predicted label', fontsize=28, fontweight="bold")
    title = "Conf Matrix"
    plt.title(title, fontsize=32, fontweight="bold")
    plt.show()


class FullyConnectedClass:
    """ The FullyConnectedClass implements a single-hidden layer neural network.
        It takes as input:

        - datasetPathTrain: the dataset used for training and validation (this is mandatory)
        - datasetPathTest: the dataset used for testing (it's not mandatory, default = None)
        - **kwargs: the hyper-parameters of the training procedure

        The hyper-parameters are:

        - _neu: number of neurons in the hidden layer (default = 100)
        - _lam: l2 regularizer used during the training procedure (default = 0.1)
        - _batch_size: the size of mini-batches for training (default = 64)
        - _epochs: number of maximum training iterations (default = 100)
        - _patience: number of epochs with no improvement in the monitored metric after which training will be stopped. Used for implementing the early stop criterion (default = 8)
        - _seed: seed set for the initialization of the weights in the network (default = 666)
        - _rolls: number of trials with the same neuron and l2 regularizer values to avoid a bad initialization of the weights (default = 1)
        - _learning_rate: how much to update the weights in response to estimated error (default = 10e-3)
        - _manager: pass the enlighten manager to keep it open from the caller of the class (default = None)

        The class provides also four public methods:

        - datasets_preparation: it randomly splits the dataset loaded from datasetPathTrain in training and validation splits using 30% of data for validation. The function normalizes the data and save the scaler in a pickle file in the LOGS folder. The file will be used for normalizing the test dataset. If datasetPathTest is not None, the function also creates the test split used for testing and normalizes them.
        - define_hyperparameters: it sets the hyperparameters as defined in kwargs argument
        - training_model: train the fully connected network and save the best model. The best model is chosen by evaluating the accuracy on the validation set

        And one private method:

        - create_model: method used to create a keras fully connected model with one hidden layer. It is called by the training_model function.
    """
    def __init__(self,
                 datasetPathTrain,
                 datasetPathTest=None,
                 **kwargs):
        super().__init__()
        # dataset splits initialization
        self.__X_train = None # training set
        self.__y_train = None # training set labels
        self.__X_val = None  # validation set
        self.__y_val = None # validation set labels
        self.__X_test = None # test set
        self.__y_test = None # test set labels
        self.__n_classes = None # number of classes
        # default hyper-parameters
        self.__neu = [100] # number of neurons
        self.__lam = [0.1] # regularization term
        self.__batch_size = 64 # batch size
        self.__epochs = 100 # number of epochs
        self.__patience = 10 # patience for early stop
        self.__seed = 666 # seed for weights extraction
        self.__rolls = 1 # number of trials
        self.__learning_rate = 0.001 # learning rate
        # progress bar manager parameters
        self.__manager = None # manager of the progress bars
        self.__close_man = True  # flag to close the progress bar
        self.__rolls_mana = None  # progress bar on the trials
        self.__neu_mana = None # progress bar on the neurons
        self.__lam_mana = None # progress bar on the regularization term

        # called functions by the class initialization
        self.datasets_preparation(datasetPathTrain, datasetPathTest)
        self.define_hyperparameters(kwargs)

    def datasets_preparation(self, datasetPathTrain, datasetPathTest):
        """Function for preparing the dataset splits. It takes as input:

        - datasetPathTrain: the dataset used for extracting training and validation sets
        - datasetPathTest: the dataset used for testing (this dataset could be None)

        The function assigns to the private variables the number of classes, the training and validation sets, the training and validation labels. If the test set is available it also assigns the test set and its labels
        """
        assert os.path.exists(datasetPathTrain), "TRAINING PATH DOES NOT EXIST"
        X_, y_ = extract_feature(datasetPathTrain) # extract the dataset and the labels computing the features
        n_classes = len(np.unique(y_)) # find the number of classes from the labels
        self.__n_classes = n_classes  # save the number of classes in the private variable of the class
        # train and validation splits with stratification, 70% is used for training and 30% for validation
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, stratify=y_, test_size=0.3, random_state=666)
        scaler = MinMaxScaler()  # initialize the scaler for normalization between 0 1
        scaler.fit(X_train)  # fit the scaler to the training data
        if not os.path.exists("../LOGS"): # check if the LOGS folder exists
            os.mkdir("../LOGS") # if not create it
        pickle.dump(scaler, open("../LOGS/FC_scaler.pkl", "wb"))  # save the scaler for testing
        X_train_norm = scaler.transform(X_train) # apply the scaler on training data
        X_val_norm = scaler.transform(X_val) # and on the validation set
        self.__X_train = X_train_norm # save the normalized training data into the class private variable
        self.__y_train = tf.keras.utils.to_categorical(y_train, n_classes) # transform the training labels into categorical labels and save them in the private variable
        self.__X_val = X_val_norm # save the normalized validation data into the class private variable
        self.__y_val = tf.keras.utils.to_categorical(y_val, n_classes) # transform the validation labels into categorical labels and save them in the private variable
        if datasetPathTest is not None:  # if the test set is available
            assert os.path.exists(datasetPathTest), "TEST PATH DOES NOT EXIST"
            X_test, y_test = extract_feature(datasetPathTest) # compute the features for the test set
            X_test_norm = scaler.transform(X_test) # normalize the data by using the scaler fitted with the training data
            self.__X_test = X_test_norm # save the normalized testing data into the class private variable
            self.__y_test = tf.keras.utils.to_categorical(y_test, n_classes) # transform the testing labels into categorical labels and save them in the private variable

    def define_hyperparameters(self, kwargs):
        """The function assigns the values of the hyperparameters based on the kwargs input.
         The hyper-parameters are:

        - _neu: number of neurons in the hidden layer (default = 100)
        - _lam: l2 regularizer used during the training procedure (default = 0.1)
        - _batch_size: the size of mini-batches for training (default = 64)
        - _epochs: number of maximum training iterations (default = 100)
        - _patience: number of epochs with no improvement in the monitored metric after which training will be stopped. Used for implementing the early stop criterion (default = 8)
        - _seed: seed set for the initialization of the weights in the network (default = 666)
        - _rolls: number of trials with the same neuron and l2 regularizer values to avoid a bad initialization of the weights (default = 1)
        - _learning_rate: how much to update the weights in response to estimated error (default = 10e-3)
        - _manager: pass the enlighten manager to keep it open from the caller of the class (default = None)

        All the hyper-parameters are assigned to the corresponding private variables
        """
        # extract the neu parameter from kwargs if it is not None
        neu = kwargs.get('_neu', None)
        if neu is not None:
            self.__neu = neu
        # extract the lam parameter from kwargs if it is not None
        lam = kwargs.get('_lam', None)
        if lam is not None:
            self.__lam = lam
        # extract the batch size parameter from kwargs if it is not None
        batch_size = kwargs.get("_batch_size", None)
        if batch_size is not None:
            self.__batch_size = int(batch_size)
        # extract the epochs parameter from kwargs if it is not None
        epochs = kwargs.get("_epochs", None)
        if epochs is not None:
            self.__epochs = int(epochs)
        # extract the patience parameter from kwargs if it is not None
        patience = kwargs.get("_patience", None)
        if patience is not None:
            self.__patience = int(patience)
        # extract the seed parameter from kwargs if it is not None
        seed = kwargs.get("_seed", None)
        if seed is not None:
            self.__seed = int(seed)
        # extract the rolls parameter from kwargs if it is not None
        rolls = kwargs.get("_rolls", None)
        if rolls is not None:
            self.__rolls = int(rolls)
        # extract the learning_rate parameter from kwargs if it is not None
        learning_rate = kwargs.get("_learning_rate", None)
        if learning_rate is not None:
            self.__learning_rate = learning_rate
        # extract the manager parameter from kwargs if it is not None
        manager = kwargs.get("_manager", None)
        if manager is not None:
            self.__manager = manager
            self.__close_man = False
        else:
            self.__close_man = True
            self.__manager = enlighten.get_manager()
        # initialize the manager parameters for the progress bars on rolls, neurons, and regularizer
        self.__rolls_mana = self.__manager.counter(total=self.__rolls, desc="Rolls", unit="num", color="blue", leave=False)
        self.__neu_mana = self.__manager.counter(total=len(self.__neu), desc="Neurons", unit="num", color="green", leave=False)
        self.__lam_mana = self.__manager.counter(total=len(self.__lam), desc="Lambda", unit="num", color="yellow", leave=False)

    def training_model(self):
        """Function to train and find the best model by evaluating the accuracy on the validation set. The function contains three for loops over the neurons, lambdas, and rolls hyperparameters. The best model will be saved in the LOGS file"""
        print("TRAINING FULLY CONNECTED")
        score_best = 0 # initialization of the best accuracy for chosing the best model
        num_feat = self.__X_train.shape[1]  # extract the number of features
        # early stop criterion during the training phase
        callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.__patience)]

        self.__neu_mana.count = 0 # initialization of the neurons progress bar
        for nn in self.__neu: # for loop on the neurons
            print("CURRENT NEURONS: {}".format(nn))
            self.__neu_mana.update() # update the neurons progress bar
            self.__lam_mana.count = 0 # initialization of the regularization term progress bar
            for la in self.__lam: # for loop on the regularization term
                self.__lam_mana.update() # update the regularization term progress bar
                self.__rolls_mana.count = 0 # initialization of the trials progress bar
                for ro in range(self.__rolls): # for loop on the trials
                    self.__rolls_mana.update() # update the trials progress bar
                    seed = self.__seed + ro * 123 # change the seed for the weights extraction
                    model = self.__create_model(num_feat=num_feat, neurons=nn, lam=la, seed=seed) # call the function to create the fully connected model
                    # train the model on the training set and evaluate on the validation set, setting the number of epochs, the batch size, and the callbacks for the early stop criterion
                    hist = model.fit(self.__X_train, self.__y_train, validation_data=(self.__X_val, self.__y_val),
                                     epochs=self.__epochs, batch_size=self.__batch_size, verbose=0,
                                     callbacks=callbacks_list)
                    score = hist.history["val_accuracy"][-1] # extract the validation accuracy
                    # if the current score is greater than the previous best, save the current score and the current model as bests
                    if score > score_best:
                        score_best = score
                        model.save('../LOGS/FC_model_best.h5')
        self.__rolls_mana.close() # close the rolls manager
        self.__neu_mana.close() # close the neurons manager
        self.__lam_mana.close() # close the lambda manager
        if self.__close_man: # close the manager if the flag of closing the manager is true
            self.__manager.stop()

    def __create_model(self, num_feat, neurons, lam, seed):
        """This function creates the fully-connected model based on the input parameters:

        - num_feat: the number of features of the data
        - neurons: the number of neurons in the hidden layer
        - lam: the value of the regularization term
        - seed: seed used to extract the values of the weights from a uniform distribution

        The function returns the created model"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate) # Adam optimizer needed for training the model
        input_shape = (num_feat,) # set the input shape as the number of features
        inp = tf.keras.layers.Input(shape=input_shape) # define the keras input layer
        # define the hidden layer having neurons neuron, relu activation function, l2 regularization term lambda, and the weights randomly initialized between -1 and 1
        x = tf.keras.layers.Dense(neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=seed),
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=seed))(inp)
        x = tf.keras.layers.Dense(self.__n_classes, activation='softmax')(x) # define the output layer having n_classes neuron and softmax function to predict the label
        model = tf.keras.models.Model(inputs=[inp], outputs=[x]) # create the model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # compile the model setting the loss function, optimizer, and the metrics to be evaluated
        return model

    def test_model(self, datasetPathTest=None):
        """ This function tests the trained model. If the datasetPathTest is None, the function check if a test set exists"""
        print("TESTING FULLY CONNECTED")
        modPath = '../LOGS/FC_model_best.h5' # path of the best model
        assert os.path.exists(modPath), "THE FULLY CONNECTED MODEL DOES NOT EXIST" # check i the model exists
        model = tf.keras.models.load_model(modPath) # load the model
        if datasetPathTest is None: # if the test path is None check if the test dataset has been already loaded
            assert self.__X_test is not None, "TEST DATASET DOES NOT EXIST"
            X_test = self.__X_test
            y_test = self.__y_test
        else: # if the datasetPathTest is not None, check if it exists, and load the data for testing
            assert os.path.exists(datasetPathTest), "TEST PATH DOES NOT EXIST"
            X, y = extract_feature(datasetPathTest) # extract the features from the data
            scaler = pickle.load(open("../LOGS/FC_scaler.pkl", "rb")) # load the scaler to normalize the data
            X_test = scaler.transform(X) # normalize the data
            y_test = tf.keras.utils.to_categorical(y, self.__n_classes) # transform the labels in categorical
        _, score = model.evaluate(X_test, y_test, verbose=0, batch_size=1) # evaluate the model on the test set
        y_pred = model.predict(X_test) # predict the labels for the confusion matrix
        print("TEST SCORE FC: ", score)
        # visualize the confusion matrix
        VisualizeConfMatrix(self.__n_classes, true_labels=np.argmax(y_test, axis=1), y_pred=np.argmax(y_pred, axis=1))


class KSVMClass:
    """ The KSVMClass implements a kernel support vector machine.
            It takes as input:

            - datasetPathTrain: the dataset used for training and validation (this is mandatory)
            - datasetPathTest: the dataset used for testing (it's not mandatory, default = None)
            - **kwargs: the hyper-parameters of the training procedure

            The hyper-parameters are:

            - _ker: the kernel adopted to project the data in a new space
            - _lam: l2 regularizer used during the training procedure (default = 0.1)
            - _gamma: the hyperparameter of the kernel. It is ignored if the kernel is linear
            - _manager: pass the enlighten manager to keep it open from the caller of the class (default = None)

            The class provides also four public methods:

            - datasets_preparation: it randomly splits the dataset loaded from datasetPathTrain in training and validation splits using 30% of data for validation. The function normalizes the data and save the scaler in a pickle file in the LOGS folder. The file will be used for normalizing the test dataset. If datasetPathTest is not None, the function also creates the test split used for testing and normalizes them.
            - define_hyperparameters: it sets the hyperparameters as defined in kwargs argument
            - training_model: train the KSVM and save the best model. The best model is chosen by evaluating the accuracy on the validation set

            And one private method:

            - create_model: method used to create a KSVM through the sklearn library. It is called by the training_model function.
        """
    def __init__(self,
                 datasetPathTrain,
                 datasetPathTest=None,
                 **kwargs):
        # dataset splits initialization
        self.__X_train = None  # training set
        self.__y_train = None  # training set labels
        self.__X_val = None  # validation set
        self.__y_val = None  # validation set labels
        self.__X_test = None  # test set
        self.__y_test = None  # test set labels
        self.__n_classes = None  # number of classes
        # default hyper-parameters
        self.__ker = 'linear'
        self.__lam = [0.1]
        self.__gam = ['auto']
        # progress bar manager parameters
        self.__manager = None  # manager of the progress bars
        self.__close_man = True  # flag to close the progress bar
        self.__gam_mana = None # progress bar on the kernel hyperparameter
        self.__lam_mana = None # progress bar on the regularization term
        # called functions by the class initialization
        self.datasets_preparation(datasetPathTrain, datasetPathTest)
        self.define_hyperparameters(kwargs)

    def datasets_preparation(self, datasetPathTrain, datasetPathTest):
        """
        Function for preparing the dataset splits. It takes as input:

        - datasetPathTrain: the dataset used for extracting training and validation sets
        - datasetPathTest: the dataset used for testing (this dataset could be None)

        The function assigns to the private variables the number of classes, the training and validation sets, the training and validation labels. If the test set is available it also assigns the test set and its labels
        """
        assert os.path.exists(datasetPathTrain), "TRAINING PATH DOES NOT EXIST"
        X_, y_ = extract_feature(datasetPathTrain) # extract the dataset and the labels computing the features
        n_classes = len(np.unique(y_)) # find the number of classes from the labels
        self.__n_classes = n_classes # save the number of classes in the private variable of the class
        # train and validation splits with stratification, 70% is used for training and 30% for validation
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, stratify=y_, test_size=0.3, random_state=666)
        scaler = MinMaxScaler() # initialize the scaler for normalization between 0 1
        scaler.fit(X_train) # fit the scaler to the training data
        if not os.path.exists("../LOGS"):  # check if the LOGS folder exists
            os.mkdir("../LOGS")  # if not create it
        pickle.dump(scaler, open("../LOGS/KSVM_scaler.pkl", "wb")) # save the scaler for testing
        X_train_norm = scaler.transform(X_train)  # apply the scaler on training data
        X_val_norm = scaler.transform(X_val)  # and on the validation set
        self.__X_train = X_train_norm  # save the normalized training data into the class private variable
        self.__y_train = y_train  # save the training in the private variable
        self.__X_val = X_val_norm  # save the normalized validation data into the class private variable
        self.__y_val = y_val  # save the validation labels in the private variable
        if datasetPathTest is not None:  # if the test set is available
            assert os.path.exists(datasetPathTest), "TEST PATH DOES NOT EXIST"
            X_test, y_test = extract_feature(datasetPathTest)  # compute the features for the test set
            X_test_norm = scaler.transform(X_test)  # normalize the data by using the scaler fitted with the training data
            self.__X_test = X_test_norm  # save the normalized testing data into the class private variable
            self.__y_test = y_test  # save the test labels in the private variable


    def define_hyperparameters(self, kwargs):
        """
        The function assigns the values of the hyperparameters based on the kwargs input.
        The hyper-parameters are:
            - _ker: the kernel adopted to project the data in a new space (default = 'linear')
            - _lam: l2 regularizer used during the training procedure (default = 0.1)
            - _gamma: the hyperparameter of the kernel. It is ignored if the kernel is linear (default='auto')
            - _manager: pass the enlighten manager to keep it open from the caller of the class (default = None)

            All the hyper-parameters are assigned to the corresponding private variables
        """
        # extract the ker parameter from kwargs if it is not None
        ker = kwargs.get('_ker', None)
        if ker is not None:
            self.__ker = ker
        # extract the lam parameter from kwargs if it is not None
        lam = kwargs.get('_lam', None)
        if lam is not None:
            self.__lam = lam
        # extract the lam parameter from kwargs if it is not None
        gam = kwargs.get('_gam', None)
        if gam is not None:
            self.__gam = gam
        # extract the manager parameter from kwargs if it is not None
        manager = kwargs.get("_manager", None)
        if manager is not None:
            self.__manager = manager
            self.__close_man = False
        else:
            self.__close_man = True
            self.__manager = enlighten.get_manager()
        # initialize the manager parameters for the progress bars on gamma and regularizer
        self.__gam_mana = self.__manager.counter(total=len(self.__gam), desc="Gamma", unit="num", color="cyan", leave=False)
        self.__lam_mana = self.__manager.counter(total=len(self.__lam), desc="Lambda", unit="num", color="gray", leave=False)

    def training_model(self):
        """Function to train and find the best model by evaluating the accuracy on the validation set. The function contains two for loops over the gammas and lambdas hyperparameters. The best model will be saved in the LOGS file"""
        print("TRAINING KSVM")
        score_best = 0 # initialization of the best accuracy for choosing the best model
        svc_best = None # initialization of the best model
        self.__gam_mana.count = 0 # initialization of the gamma progress bar
        for ga in self.__gam: # for loop on the gammas. If ker = 'linear' the for will loop only once
            self.__gam_mana.update() # update the gammas progress bar
            self.__lam_mana.count = 0 # initialization of the lambda progress bar
            for la in self.__lam: # for loop on the lambdas
                self.__lam_mana.update() # update the lambdas progress bar
                svc = SVC(kernel=self.__ker, gamma=ga, C=la, decision_function_shape='ovo') # create the KSVM model to be trained, using one-vs-one strategy for multi-class problems
                svc.fit(self.__X_train, self.__y_train) # train the model on the training data
                score = svc.score(self.__X_val, self.__y_val) # compute the validation score to find the best model
                # if the current score is greater than the previous best, save the current score and the current model as bests
                if score > score_best:
                    score_best = score
                    svc_best = svc
        self.__gam_mana.close() # close the gammas manager
        self.__lam_mana.close() # close the lambda manager
        if self.__close_man: # close the manager if the flag of closing the manager is true
            self.__manager.stop()
        # write the best model on a pickle file
        filename = '../LOGS/SVC_best'
        pickle.dump(svc_best, open(filename, 'wb'))

    def test_model(self, datasetPathTest=None):
        print("TESTING KSVM")
        modPath = '../LOGS/SVC_best' # path of the best model
        assert os.path.exists(modPath), "THE KSVM DOES NOT EXIST"  # check if the model exists
        svc = pickle.load(open(modPath, 'rb')) # load the best model
        if datasetPathTest is None:  # if the test path is None check if the test dataset has been already loaded
            assert self.__X_test is not None, "TEST DATASET DOES NOT EXIST"
            X_test = self.__X_test
            y_test = self.__y_test
        else:  # if the datasetPathTest is not None, check if it exists, and load the data for testing
            assert os.path.exists(datasetPathTest), "TEST PATH DOES NOT EXIST"
            X, y = extract_feature(datasetPathTest)  # extract the features from the data
            scaler = pickle.load(open("../LOGS/KSVM_scaler.pkl", "rb"))  # load the scaler to normalize the data
            X_test = scaler.transform(X)  # normalize the data
            y_test = y
        score = svc.score(X_test, y_test) # evaluate the model on the test set
        print("TEST SCORE KSVM: ", score)
        y_pred = svc.predict(X_test) # predict the labels to plot the confusion matrix
        VisualizeConfMatrix(self.__n_classes, y_test, y_pred)



class CNN1DClass:
    """ The CNN1DClass implements a 1D convolutional neural network.
            It takes as input:

            - datasetPathTrain: the dataset used for training and validation (this is mandatory)
            - datasetPathTest: the dataset used for testing (it's not mandatory, default = None)
            - **kwargs: the hyper-parameters of the training procedure

            The hyper-parameters are:

            - _filters: the number of filters in each convolutionl layer (default = [(8, 8, 8)], i.e. 3 conv layers with 8 filters each)
            - _kernels: the kernel size applied to each conv layer (default = [8])
            - _batch_size: the size of mini-batches for training (default = 64)
            - _epochs: number of maximum training iterations (default = 100)
            - _patience: number of epochs with no improvement in the monitored metric after which training will be stopped. Used for implementing the early stop criterion (default = 8)
            - _learning_rate: how much to update the weights in response to estimated error (default = 10e-3)
            - _manager: pass the enlighten manager to keep it open from the caller of the class (default = None)

            The class provides also four public methods:

            - datasets_preparation: it randomly splits the dataset loaded from datasetPathTrain in training and validation splits using 30% of data for validation. The function normalizes the data and save the scaler in a pickle file in the LOGS folder. The file will be used for normalizing the test dataset. If datasetPathTest is not None, the function also creates the test split used for testing and normalizes them.
            - define_hyperparameters: it sets the hyperparameters as defined in kwargs argument
            - training_model: train the fully connected network and save the best model. The best model is chosen by evaluating the accuracy on the validation set

            And one private method:

            - create_model: method used to create a keras 1D convolutional neural network. It is called by the training_model function.
        """
    def __init__(self,
                 datasetPathTrain,
                 datasetPathTest=None,
                 **kwargs):
        # dataset splits initialization
        self.__X_train = None  # training set
        self.__y_train = None  # training set labels
        self.__X_val = None  # validation set
        self.__y_val = None  # validation set labels
        self.__X_test = None  # test set
        self.__y_test = None  # test set labels
        self.__n_classes = None  # number of classes
        # default hyper-parameters
        self.__filters = [(8, 8, 8)] # number of filters
        self.__kernel = [8] # kernel size
        self.__batch_size = 64  # batch size
        self.__epochs = 100  # number of epochs
        self.__patience = 8  # patience for early stop
        self.__learning_rate = 0.001 # learning rate
        # progress bar manager parameters
        self.__manager = None  # manager of the progress bars
        self.__close_man = True  # flag to close the progress bar
        self.__filters_mana = None
        self.__kernel_mana = None
        # called functions by the class initialization
        self.datasets_preparation(datasetPathTrain, datasetPathTest)
        self.define_hyperparameters(kwargs)


    def datasets_preparation(self, datasetPathTrain, datasetPathTest):
        """Function for preparing the dataset splits. It takes as input:

               - datasetPathTrain: the dataset used for extracting training and validation sets
               - datasetPathTest: the dataset used for testing (this dataset could be None)

           The function assigns to the private variables the number of classes, the training and validation sets, the training and validation labels. If the test set is available it also assigns the test set and its labels
        """
        assert os.path.exists(datasetPathTrain), "TRAINING PATH DOES NOT EXIST"
        X_, y_ = load_dataset(datasetPathTrain) # load the dataset and the labels
        n_classes = len(np.unique(y_)) # find the number of classes from the labels
        self.__n_classes = n_classes # save the number of classes in the private variable of the class
        # train and validation splits with stratification, 70% is used for training and 30% for validation
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, stratify=y_, test_size=0.3, random_state=666)
        scaler = MinMaxScaler() # initialize the scaler
        # reshape the data to normalize along the channels
        X_train_resh = np.reshape(X_train, [X_train.shape[0] * X_train.shape[1], X_train.shape[2]])
        X_val_resh = np.reshape(X_val, [X_val.shape[0] * X_val.shape[1], X_val.shape[2]])
        scaler.fit(X_train_resh) # fit the scaler to the training data
        if not os.path.exists("../LOGS"):  # check if the LOGS folder exists
            os.mkdir("../LOGS")  # if not create it
        pickle.dump(scaler, open("../LOGS/CNN_scaler.pkl", "wb")) # save the scaler for testing
        X_train_norm = scaler.transform(X_train_resh) # apply the scaler on training data
        X_val_norm = scaler.transform(X_val_resh) # and on the validation set
        self.__X_train = np.reshape(X_train_norm, [X_train.shape[0], X_train.shape[1], X_train.shape[2]]) # save the normalized training data into the class private variable reshaping them back to the original shape
        self.__X_val = np.reshape(X_val_norm, [X_val.shape[0], X_val.shape[1], X_val.shape[2]]) # save the normalized validation data into the class private variable reshaping them back to the original shape
        self.__y_train = tf.keras.utils.to_categorical(y_train, self.__n_classes) # transform the training labels into categorical labels and save them in the private variable
        self.__y_val = tf.keras.utils.to_categorical(y_val, self.__n_classes) # transform the validation labels into categorical labels and save them in the private variable
        if datasetPathTest is not None:  # if the test set is available
            assert os.path.exists(datasetPathTest), "TEST PATH DOES NOT EXIST"
            X_test, y_test = load_dataset(datasetPathTest)  # load the test set
            X_test_resh = np.reshape(X_test, [X_test.shape[0] * X_test.shape[1], X_test.shape[2]]) # reshape the testing data to apply the normalization along the channels
            X_test_norm = scaler.transform(X_test_resh)  # normalize the data by using the scaler fitted with the training data
            self.__X_test = np.reshape(X_test_norm, [X_test.shape[0], X_test.shape[1], X_test.shape[2]])  # save the normalized testing data into the class private variable reshaping them back to the original shape
            self.__y_test = tf.keras.utils.to_categorical(y_test, n_classes)  # transform the testing labels into categorical labels and save them in the private variable


    def define_hyperparameters(self, kwargs):
        """The function assigns the values of the hyperparameters based on the kwargs input.
                 The hyper-parameters are:

                - _filters: the number of filters in each convolutionl layer (default = [(8, 8, 8)], i.e. 3 conv layers with 8 filters each)
                - _kernels: the kernel size applied to each conv layer (default = [8])
                - _batch_size: the size of mini-batches for training (default = 64)
                - _epochs: number of maximum training iterations (default = 100)
                - _patience: number of epochs with no improvement in the monitored metric after which training will be stopped. Used for implementing the early stop criterion (default = 8)
                - _learning_rate: how much to update the weights in response to estimated error (default = 10e-3)
                - _manager: pass the enlighten manager to keep it open from the caller of the class (default = None)

                All the hyperparameters are assigned to the corresponding private variables
        """
        # extract the filters parameter from kwargs if it is not None
        filters = kwargs.get("_filters", None)
        if filters is not None:
            self.__filters = filters
        # extract the kernel parameter from kwargs if it is not None
        kernel = kwargs.get("_kernels", None)
        if kernel is not None:
            self.__kernel = kernel
        # extract the batch size parameter from kwargs if it is not None
        batch_size = kwargs.get("_batch_size", None)
        if batch_size is not None:
            self.__batch_size = int(batch_size)
        # extract the epochs parameter from kwargs if it is not None
        epochs = kwargs.get("_epochs", None)
        if epochs is not None:
            self.__epochs = int(epochs)
        # extract the patience parameter from kwargs if it is not None
        patience = kwargs.get("_patience", None)
        if patience is not None:
            self.__patience = int(patience)
        # extract the learning_rate parameter from kwargs if it is not None
        learning_rate = kwargs.get("_learning_rate", None)
        if learning_rate is not None:
            self.__learning_rate = learning_rate
        # extract the manager parameter from kwargs if it is not None
        manager = kwargs.get("_manager", None)
        if manager is not None:
            self.__manager = manager
            self.__close_man = False
        else:
            self.__close_man = True
            self.__manager = enlighten.get_manager()
        # initialize the manager parameters for the progress bars on filters and kernel
        self.__filters_mana = self.__manager.counter(total=len(self.__filters), desc="Filters", unit="num", color="blue", leave=False)
        self.__kernel_mana = self.__manager.counter(total=len(self.__kernel), desc="Kernels", unit="num", color="green", leave=False)

    def training_model(self):
        """Function to train and find the best model by evaluating the accuracy on the validation set. The function contains three for loops over the neurons, lambdas, and rolls hyperparameters. The best model will be saved in the LOGS file"""
        print("TRAINING 1D CNN")
        callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.__patience)] # early stop criterion during the training phase
        score_best = 0 # initialization of the best accuracy for choosing the best model
        self.__filters_mana.count = 0 # initialization of the filters progress bar
        for fil in self.__filters: # for loop on the filters
            self.__filters_mana.update() # update the filters progress bar
            self.__kernel_mana.count = 0 # initialization of the kernels progress bar
            for ker in self.__kernel: # for loop on the kernels
                self.__kernel_mana.update() # update the kernels progress bar
                model = self.create_model(filters=fil, kernels=ker) # call the function to create the 1d cnn model
                # train the model on the training set and evaluate on the validation set, setting the number of epochs, the batch size, and the callbacks for the early stop criterion
                hist = model.fit(self.__X_train, self.__y_train, validation_data=(self.__X_val, self.__y_val),
                                 epochs=self.__epochs, batch_size=self.__batch_size, verbose=0, callbacks=callbacks_list)
                score = hist.history["val_accuracy"][-1] # extract the validation accuracy
                # if the current score is greater than the previous best, save the current score and the current model as bests
                if score > score_best:
                    score_best = score
                    model.save('../LOGS/CNN_model_best.h5')
        self.__filters_mana.close() # close the filters manager
        self.__kernel_mana.close() # close the kernel manager
        if self.__close_man: # close the manager if the flag of closing the manager is true
            self.__manager.stop()

    def create_model(self, filters, kernels):
        """This function creates the fully-connected model based on the input parameters:

                - filters: the number of filters in each con layer
                - kernels: the kernel size

            The function returns the created model"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)  # Adam optimizer needed for training the model
        input_shape = self.__X_train.shape
        inp = tf.keras.layers.Input(shape=input_shape[1:]) # initialize the input layer based on the input size
        x = None
        # the 1D CNN is made of n functional blocks: 1D conv layer, dropout, and average pooling. the number of blocks n is equal to the number of filters.
        # After the n blocks, a Dense layer with 10 neurons and relu activation function is stacked. The last dense layer predicts the label through the sofmax function.
        for i, f in enumerate(filters):
            if i == 0:
                x = tf.keras.layers.Conv1D(filters=f, kernel_size=kernels, activation='relu', padding="same")(inp)
            else:
                x = tf.keras.layers.Conv1D(filters=f, kernel_size=kernels, activation='relu', padding="same")(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.AveragePooling1D(pool_size=2, padding="valid")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        x = tf.keras.layers.Dense(self.__n_classes, activation='softmax')(x) # define the output layer having n_classes neuron and softmax function to predict the label
        model = tf.keras.models.Model(inputs=[inp], outputs=[x]) # create the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # compile the model setting the loss function, optimizer, and the metrics to be evaluated
        return model

    def test_model(self, datasetPathTest=None):
        """ This function tests the trained model. If the datasetPathTest is None, the function check if a test set exists"""
        print("TESTING CNN")
        modPath = '../LOGS/CNN_model_best.h5'  # path of the best model
        assert os.path.exists(modPath), "THE CNN MODEL DOES NOT EXIST"  # check if the model exists
        model = tf.keras.models.load_model(modPath) # load the model
        if datasetPathTest is None:  # if the test path is None check if the test dataset has been already loaded
            assert self.__X_test is not None, "TEST DATASET DOES NOT EXIST"
            X_test = self.__X_test
            y_test = self.__y_test
        else: # if the datasetPathTest is not None, check if it exists, and load the data for testing
            assert os.path.exists(datasetPathTest), "TEST PATH DOES NOT EXIST"
            X, y = load_dataset(datasetPathTest) # load the test set
            scaler = pickle.load(open("../LOGS/CNN_scaler.pkl", "rb")) # load the scaler to normalize the data
            X_test_resh = np.reshape(X, [X.shape[0] * X.shape[1], X.shape[2]]) # reshape the data to normalize them along the channels
            X_test_norm = scaler.transform(X_test_resh) # normalize the data
            X_test = np.reshape(X_test_norm, [X.shape[0], X.shape[1], X.shape[2]]) # reshape back the data to the original shape
            y_test = tf.keras.utils.to_categorical(y, self.__n_classes) # transform the labels in categorical
        _, score = model.evaluate(X_test, y_test, verbose=0, batch_size=1)  # evaluate the model on the test set
        y_pred = model.predict(X_test)  # predict the labels for the confusion matrix
        print("TEST SCORE CNN: ", score)
        # visualize the confusion matrix
        VisualizeConfMatrix(self.__n_classes, true_labels=np.argmax(y_test, axis=1), y_pred=np.argmax(y_pred, axis=1))
        