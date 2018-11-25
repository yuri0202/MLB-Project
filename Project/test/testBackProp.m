% Script per testare la back propagation

% Scelta dei parametri per la rete neurale
SUP_WEIGHTS = 0.09;
INF_WEIGHTS = -0.09;
OUTPUT_ACTIVATION_FUNCTION = @identity;
OUTPUT_ACTIVATION_FUNCTION_DX =@identityDx;
HIDDEN_ACTIVATION_FUNCTION = @sigmoid;
HIDDEN_ACTIVATION_FUNCTION_DX = @sigmoidDx;
INPUT_DIMENSION = 4;
OUTPUT_DIMENSION = 2;
% Creo la rete neurale
net = createNeuralNetwork(INPUT_DIMENSION, OUTPUT_DIMENSION, OUTPUT_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION_DX, [
    struct('size',3,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer1
    struct('size',2,'function',HIDDEN_ACTIVATION_FUNCTION,'derivative',HIDDEN_ACTIVATION_FUNCTION_DX) % Hidden Layer2
],INF_WEIGHTS,SUP_WEIGHTS );


input = [1,1,3,4; 3 4 2 2; 1 4 4 2];
target = [1;2;4];

[outputs, A] = forwardProp(net, input, true);

[dw,db] = backProp(net,input,outputs,A,target,@crossEntropyDx);

