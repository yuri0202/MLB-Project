% Script per testare la RProp
clc;
clear;

[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

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

etaMins = [0.4 0.5 0.6];
etaPlus = [1.1 1.2 1.3];
numHiddenNodes = [50 60 70];
meanStdPerComb = modelHyperParametersOptimization(trainImages,trainLabels,500,EPOCHS,OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, HIDDEN_ACTIVATION_FUNCTION, HIDDEN_ACTIVATION_FUNCTION_DX, SUP_WEIGHTS, INF_WEIGHTS,ERROR_FUNCTION ,ERROR_FUNCTION_DX,SOFTMAX_FLAG,K,etaMins,etaPlus,numHiddenNodes,PRINT_FLAG);