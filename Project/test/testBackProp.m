% Script per testare la back propagation

% Scelta dei parametri per la rete neurale
SUP_WEIGHTS = 0.09;
INF_WEIGHTS = -0.09;
OUTPUT_ACTIVATION_FUNCTION = @identity;
OUTPUT_ACTIVATION_FUNCTION_DX =@identityDx;
HIDDEN_ACTIVATION_FUNCTION = @sigmoid;
HIDDEN_ACTIVATION_FUNCTION_DX = @sigmoidDx;
INPUT_DIMENSION = 2;
OUTPUT_DIMENSION = 1;
ERROR_FUNCTION_DX = @sumOfSquaresDx;
% Creo la rete neurale
net = createNeuralNetwork(INPUT_DIMENSION, OUTPUT_DIMENSION, OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, [
    struct('size',8,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer1
],INF_WEIGHTS,SUP_WEIGHTS );

input = [1,2;2,1;0,1];
target = [1;1;0];

% Eseguo la propagazione in avanti
[outputs, A] = forwardProp(net, input, true);

% Eseguo la back-propagation
[dW,dB] = backProp(net,input,outputs,A,target,ERROR_FUNCTION_DX);

