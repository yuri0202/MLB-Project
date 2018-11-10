function [derivativeWeights, derivativeBiases] = backProp(net, input, outputs, A, target)
%BACKPROP Esegue la back-propagation sulla rete
%   INPUT:
%       - net: la rete neurale che verrà utilizzata.
%       - input: matrice (N x d) con N numero di campioni e
%         d numero di componenti.
%       - outputs: cell array prodotto in output dalla funzione di 
%         forward propagation.
%       - A: cell array di nodi di input prodotto dalla funzione di 
%         forward propagation.
%       - target: target per l'input dato.
%
%   OUTPUT:
%       - derivativeWeights: cell array contenente le derivate dei pesi per
%         ogni strato.
%       - derivativeBiases: cell array contenente le derivate dei bias per
%         ogni strato.
    
    %calcola i delta dei nodi di output
    layer = net.hiddenLayersNum+1;
    delta{layer} = net.errorFunctionDerivative(outputs{layer}, target) .* net.derivativeFunctions{layer}(A{layer});

    %calcola i delta rimanenti "all'indietro"
    for i=layer-1:-1:1
        delta{i} = net.derivativeFunctions{i}(A{i}) .* (delta{i+1}*net.W{i+1});
    end
    
    %calcola le derivate
    derivativeWeights{1} = delta{1}' * input;
    derivativeBiases{1} = sum(delta{1});
    
    for i=2:layer
        derivativeWeights{i} = delta{i}' * outputs{i-1};
        derivativeBiases{i} = sum(delta{i});
    end
end

