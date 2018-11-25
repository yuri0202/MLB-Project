% Script per testare la scelta degli iper-parametri tramite K-Fold Cross
% Validation
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
K = 10;
SOFTMAX_FLAG = true;
PRINT_FLAG = false;
SET_SIZE_FOR_KFOLD = 10000;
etaMins = [0.5];
etaPlus = [1.1];
numHiddenNodes = [200];

% Lancio l'algoritmo per la scelta dei migliori iper-parametri per tramite
% la tecnica di k-fold cross validation
meanStdPerComb = modelHyperParametersOptimization(trainImages,trainLabels,SET_SIZE_FOR_KFOLD,EPOCHS,OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, HIDDEN_ACTIVATION_FUNCTION, HIDDEN_ACTIVATION_FUNCTION_DX, SUP_WEIGHTS, INF_WEIGHTS,ERROR_FUNCTION ,ERROR_FUNCTION_DX,SOFTMAX_FLAG,K,etaMins,etaPlus,numHiddenNodes,PRINT_FLAG);