% Script per testare la RProp
clc;
clear;
[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

TRAINING_SET_SIZE = 15000;
VALIDATION_SET_SIZE = 7500; 
TEST_SET_SIZE = 4000;
HIDDEN_NODES = 320;
EPOCHS = 150;
OUTPUT_ACTIVATION_FUNCTION = @identity;
OUTPUT_ACTIVATION_FUNCTION_DX =@identityDx;
HIDDEN_ACTIVATION_FUNCTION = @sigmoid;
HIDDEN_ACTIVATION_FUNCTION_DX = @sigmoidDx;
ERROR_FUNCTION = @crossEntropy;
ERROR_FUNCTION_DX = @crossEntropyDx;
ETA_MINUS = 0.5;
ETA_PLUS = 1.2;
SOFTMAX_FLAG = true;
PRINT_FLAG = true;
[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = createSets(trainImages', trainLabels, testImages', testLabels, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE);




net = createNeuralNetwork(size(trainingSetData,2), 10, OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, [
    struct('size',HIDDEN_NODES,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer
]);

[net, trainingSetErrors, validationSetErrors] = trainNeuralNetworkRProp(net, trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, EPOCHS, ERROR_FUNCTION, ERROR_FUNCTION_DX, ETA_MINUS, ETA_PLUS, SOFTMAX_FLAG, PRINT_FLAG);

plotErrors(trainingSetErrors, validationSetErrors);