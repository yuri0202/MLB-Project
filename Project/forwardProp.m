function [ outputs, A ] = forwardProp( net, input )

% Questa funzione simula la propagazione in avanti degli input specificati
% su una rete neurale
%
%   INPUT:
%       - 'net':  La rete neurale da utilizzare per la propagazione
%       - 'input': Matrice di dimensione N x d, con N numero di input da dare
%   alla rete, e d è la dimensionalità dell'input
%
%   OUTPUT:
%       - 'outputs': Un cell array dove l'i-esima cella contiene l'output
%           dell'i-esimo hidden layer. L'ultima cella contiene l'output dello
%           strato di output
%       - 'A': Un array che contiene l'input di ogni nodo dell'i-esimo strato
%            per ognuno degli N input dati alla rete


    %Controllo se vi è mismatch tra l'input e il parametro inputDimension della rete
    if(net.inputDimension ~= size(input,2))
        error('forwardPropagation: Errore sulla dimensione dell''input. L''input ha dimensione %d e la dimensione prevista è %d.',size(input,2),net.inputDimension);
    end
    
    %Propagiamo l'input in avanti attraverso ognuno degli strati
    layerInput = input;
    outputs = cell(net.hiddenLayersNum + 1, 1); 
    A = cell(net.hiddenLayersNum + 1, 1); 
    for i=1 : net.hiddenLayersNum + 1 
        A{i} = (net.W{i} * layerInput')';
        B = repmat(net.b{i},size(layerInput,1),1);
        outputs{i} = net.outputFunctions{i}(A{i}+B);
        layerInput = outputs{i}; %L'output sarà l'input per la prossima propagazione
    end
end

