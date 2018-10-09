function [ outputs, A ] = forwardPropagation( net, input )
%FORWARDPROPAGATION Computes the forward propagation on net with the given input
%   INPUT:
%   - net is the neural network to use
%   - input is a matrix of N rows where each row is an input for the neural 
%     network
%   OUTPUT:
%   - outputs is a cell array where the i-th cell contains the output of 
%     the i-th hidden layer. The last cell contains the output of the 
%     output layer.
%   - A is a cell array containing the input of each node of the i-th layer
%     for each of the N samples.


    %if there's a mismatch between the input and the network's inputDimesion
    if(net.inputDimension ~= size(input,2))
        error('forwardPropagation: Input size error.\nInput dimension was %d. Expected dimension: %d.',size(input,2),net.inputDimension);
    end
    
    %we need to propagate the input forward through each layer
    layerInput = input;
    outputs = cell(net.hiddenLayersNum + 1, 1); %preallocating outputs
    A = cell(net.hiddenLayersNum + 1, 1); %preallocating A for speed
    for i=1 : net.hiddenLayersNum + 1 
        A{i} = (net.W{i} * layerInput')';
        B = repmat(net.b{i},size(layerInput,1),1);
        outputs{i} = net.outputFunctions{i}(A{i}+B);
        layerInput = outputs{i}; %the current output will be the input for the next propagation
    end
end

