from ElaborationClass import FullyConnectedClass, CNN1DClass, KSVMClass

datasetPathTrain = "../Dataset/Merged/80_Samples/train/" # training set path. It is mandatory.
datasetPathTest = "../Dataset/Merged/80_Samples/test/" # test set path. It is not mandatory when a class object is initialized.


svc = KSVMClass(datasetPathTrain=datasetPathTrain, _ker='linear', _lam=[10**i for i in range(-4, 5)])
svc.training_model()
svc.test_model(datasetPathTest)

# fc = FullyConnectedClass(datasetPathTrain=datasetPathTrain, _neu=[10*i for i in range(1, 11)],
#                          _lam=[10**i for i in range(-3, 4)], _batch_size=32, _epochs=100, _patience=8, _rolls=1)
# fc.training_model()
# fc.test_model(datasetPathTest=datasetPathTest)
#
# cnn = CNN1D(datasetPathTrain=datasetPathTrain, _filters=[(8, 8), (16, 16), (16, 32), (4, 8, 16), (8, 8, 8), (8, 16, 32), (4, 8, 16, 32)], _kernels=[8, 12, 16])
# cnn.training_model()
# cnn.test_model(datasetPathTest=datasetPathTest)

