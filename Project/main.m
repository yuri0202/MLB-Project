% Main per testare la PARTE B
clc;
clear;

% Estraggo i datasets
[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

% Definizione dei parametri per la rete neurale, l'addestramento e la
% k-fold
SUP_WEIGHTS = 0.09;
INF_WEIGHTS = -0.09;
EPOCHS = 150;
OUTPUT_ACTIVATION_FUNCTION = @identity;
OUTPUT_ACTIVATION_FUNCTION_DX =@identityDx;
HIDDEN_ACTIVATION_FUNCTION = @sigmoid;
HIDDEN_ACTIVATION_FUNCTION_DX = @sigmoidDx;
ERROR_FUNCTION = @crossEntropy;
ERROR_FUNCTION_DX = @crossEntropyDx;
TRAINING_SET_SIZE = 15000;
VALIDATION_SET_SIZE = 4000;
TEST_SET_SIZE = 4000;
K = 10;
SOFTMAX_FLAG = true;
PRINT_FLAG = false;
SET_SIZE_FOR_KFOLD = 10000;
etaMins = [0.4];
etaPlus = [1.1];
numHiddenNodes = [600,800];

% Lancio l'algoritmo per la scelta dei migliori iper-parametri per tramite
% la tecnica di k-fold cross validation
tic;
[meanStdPerComb,bestNumNodes,bestEtaMin,bestEtaPlus] = modelHyperParametersOptimization(@topAvgConsideringStd,trainImages,trainLabels,SET_SIZE_FOR_KFOLD,EPOCHS,OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, HIDDEN_ACTIVATION_FUNCTION, HIDDEN_ACTIVATION_FUNCTION_DX, SUP_WEIGHTS, INF_WEIGHTS,ERROR_FUNCTION ,ERROR_FUNCTION_DX,SOFTMAX_FLAG,K,etaMins,etaPlus,numHiddenNodes,PRINT_FLAG);
fprintf("\nTempo per l'esecuzione della K-Fold Cross Validation: %.0f minuti e %.0f secondi\n", floor(toc/60), (toc) - (floor(toc/60)*60));
[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = createSets(trainImages', trainLabels, testImages', testLabels, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE);

% Creazione rete
net = createNeuralNetwork(size(trainingSetData,2), 10, OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, [
    struct('size',bestNumNodes,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer
], INF_WEIGHTS,SUP_WEIGHTS);

% Addestrameto RProp
[net, trainingSetErrors, validationSetErrors] = trainNeuralNetworkRProp(net, trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, EPOCHS, ERROR_FUNCTION, ERROR_FUNCTION_DX, bestEtaMin, bestEtaPlus, SOFTMAX_FLAG, true);

% Grafico per gli errori sul training e sul validation
plotErrors(trainingSetErrors, validationSetErrors);

% Forward propagation sul test set
[output, ~] = forwardProp(net,testSetData, true);

% Calcolo dell'accuracy
[totalAccuracy] = evaluateNetClassifier(output{net.hiddenLayersNum+1}, testSetLabels);
fprintf("\n Accuracy della rete neurale: %.2f%% \n", totalAccuracy*100);