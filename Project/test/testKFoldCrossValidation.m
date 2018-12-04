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
K = 10;
SOFTMAX_FLAG = true;
PRINT_FLAG = false;
SET_SIZE_FOR_KFOLD = 10000;
etaMins = [0.4,0.5,0.6];
etaPlus = [1.1,1.2,1.3];
numHiddenNodes = [200,400,600,800];
SELECTION_CRITERION_FUNCTION = @topAvgConsideringStd;

% Lancio l'algoritmo per la scelta dei migliori iper-parametri tramite
% la tecnica di K-Fold Cross Validation
tic;
[meanStdPerComb,bestNumNodes,bestEtaMin,bestEtaPlus] = modelHyperParametersOptimization(SELECTION_CRITERION_FUNCTION,trainImages,trainLabels,SET_SIZE_FOR_KFOLD,EPOCHS,OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, HIDDEN_ACTIVATION_FUNCTION, HIDDEN_ACTIVATION_FUNCTION_DX, SUP_WEIGHTS, INF_WEIGHTS,ERROR_FUNCTION ,ERROR_FUNCTION_DX,SOFTMAX_FLAG,K,etaMins,etaPlus,numHiddenNodes,PRINT_FLAG);
fprintf("\nTempo per l'esecuzione della K-Fold Cross Validation: %.0f minuti e %.0f secondi\n", floor(toc/60), (toc) - (floor(toc/60)*60));