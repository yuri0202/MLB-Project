% Script per testare la RProp
clc;
clear;
[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

SUP_WEIGHTS = 0.09;
INF_WEIGHTS = -0.09;
TRAINING_SET_SIZE = 10000;
VALIDATION_SET_SIZE = 4000; 
TEST_SET_SIZE = 5000;
HIDDEN_NODES = 320;
EPOCHS = 150;
OUTPUT_ACTIVATION_FUNCTION = @identity;
OUTPUT_ACTIVATION_FUNCTION_DX =@identityDx;
HIDDEN_ACTIVATION_FUNCTION = @sigmoid;
HIDDEN_ACTIVATION_FUNCTION_DX = @sigmoidDx;
ERROR_FUNCTION = @crossEntropy;
ERROR_FUNCTION_DX = @crossEntropyDx;
ETA_MINUS = 0.6;
ETA_PLUS = 1.1;
SOFTMAX_FLAG = true;
PRINT_FLAG = true;
%[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = createSets(trainImages', trainLabels, testImages', testLabels, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE);

trainingSetData = [1 1 1 1 1 1 1 1 1 1; 2 2 2 2 2 2 2 2 2 2; 3 3 3 3 3 3 3 3 3 3; 4 4 4 4 4 4 4 4 4 4; 5 5 5 5 5 5 5 5 5 5; 6 6 6 6 6 6 6 6 6 6; 7 7 7 7 7 7 7 7 7 7; 8 8 8 8 8 8 8 8 8 8; 9 9 9 9 9 9 9 9 9 9; 10 10 10 10 10 10 10 10 10 10];
shuffle   = randperm(10);
trainingSetData   = trainingSetData(shuffle,:);

%


%net = createNeuralNetwork(size(trainingSetData,2), 10, OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, [
    %struct('size',HIDDEN_NODES,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer
%], INF_WEIGHTS,SUP_WEIGHTS);

%[net, trainingSetErrors, validationSetErrors] = trainNeuralNetworkRProp(net, trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, EPOCHS, ERROR_FUNCTION, ERROR_FUNCTION_DX, ETA_MINUS, ETA_PLUS, SOFTMAX_FLAG, PRINT_FLAG);

%plotErrors(trainingSetErrors, validationSetErrors);
etaMins = [0.4 0.5 0.6];
etaPlus = [1.1, 1.2, 1.3];
numHiddenNodes = [300 400 500];
%kFoldCrossValidation(net,10,150,@crossEntropy,@crossEntropyDx,true,10,etaMins,etaPlus,numHiddenNodes)