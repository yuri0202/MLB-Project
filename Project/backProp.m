function [derivativeWeights, derivativeBiases] = backProp(net, input, outputs, A, target, errorFunction)
%BACKPROP Esegue la back-propagation sulla rete
%   INPUT:
%       - 'net': la rete neurale che verr� utilizzata.
%       - 'input': matrice (N x d) con N numero di campioni e
%         d numero di componenti.
%       - 'outputs': cell array prodotto in output dalla funzione di 
%         forward propagation.
%       - 'A': cell array di nodi di input prodotto dalla funzione di 
%         forward propagation.
%       - 'target': target per l'input dato.
%       - 'errorFunction': Funzione da utilizzare per il calcolo
%       dell'errore
%
%   OUTPUT:
%       - derivativeWeights: cell array contenente le derivate dei pesi per
%         ogni strato.
%       - derivativeBiases: cell array contenente le derivate dei bias per
%         ogni strato.
    
    % Calcola i delta dei nodi di output
    layer = net.hiddenLayersNum+1;
    delta{layer} = errorFunction(outputs{layer}, target) .* net.derivativeFunctions{layer}(A{layer});

    % Calcola i delta rimanenti "all'indietro"
    for i=layer-1:-1:1
        delta{i} = net.derivativeFunctions{i}(A{i}) .* (delta{i+1}*net.W{i+1});
    end
    
    % Calcola le derivate
    derivativeWeights{1} = delta{1}' * input;
    derivativeBiases{1} = sum(delta{1});
    
    for i=2:layer
        derivativeWeights{i} = delta{i}' * outputs{i-1};
        derivativeBiases{i} = sum(delta{i});
    end
end

