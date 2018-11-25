% Script per testare la RProp
clc;
clear;

% Carico i dataset
[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

% Definizione dei parametri per la creazione della rete e per
% l'addestramento
SUP_WEIGHTS = 0.09;
INF_WEIGHTS = -0.09;
TRAINING_SET_SIZE = 10000;
VALIDATION_SET_SIZE = 3000; 
TEST_SET_SIZE = 3000;
HIDDEN_NODES = 300;
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

%shuffle   = randperm(TRAINING_SET_SIZE);
%trainingSetData   = trainingSetData(shuffle,:);
%trainingSetLabels = trainingSetLabels(shuffle,:);

%shuffle2   = randperm(VALIDATION_SET_SIZE);
%validationSetData   = validationSetData(shuffle2,:);
%validationSetLabels = validationSetLabels(shuffle2,:);

%shuffle3   = randperm(TEST_SET_SIZE);
%testSetData   = testSetData(shuffle3,:);
%testSetLabels = testSetLabels(shuffle3,:);

% Creazione rete
net = createNeuralNetwork(size(trainingSetData,2), 10, OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, [
    struct('size',HIDDEN_NODES,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer
], INF_WEIGHTS,SUP_WEIGHTS);

% Addestrameto RProp
[net, trainingSetErrors, validationSetErrors] = trainNeuralNetworkRProp(net, trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, EPOCHS, ERROR_FUNCTION, ERROR_FUNCTION_DX, ETA_MINUS, ETA_PLUS, SOFTMAX_FLAG, PRINT_FLAG);

% Grafico per gli errori sul training e sul validation
plotErrors(trainingSetErrors, validationSetErrors);

% Forward propagation sul test set
[output, ~] = forwardProp(net,testSetData, true);

% Calcolo dell'accuracy
[totalAccuracy] = evaluateNetClassifier(output{net.hiddenLayersNum+1}, testSetLabels);
fprintf("\n Accuracy della rete neurale: %.2f%% \n", totalAccuracy*100);