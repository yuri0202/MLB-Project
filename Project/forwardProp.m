function [ outputs, A ] = forwardProp( net, input, softMax )

% Questa funzione simula la propagazione in avanti degli input specificati
% su una rete neurale
%
%   INPUT:
%       - 'net':  La rete neurale da utilizzare per la propagazione
%       - 'input': Matrice di dimensione N x d, con N numero di input da
%                  dare alla rete e d è la dimensionalità dell'input
%   	- 'softMax': Se uguale a true, all'output della rete verrà
%                    applicato il softmax, no altrimenti 
%
%   OUTPUT:
%       - 'outputs': Un cell array dove l'i-esima cella contiene l'output
%           dell'i-esimo hidden layer. L'ultima cella contiene l'output
%           dello strato di output
%       - 'A': Un array che contiene l'input di ogni nodo dell'i-esimo
%              strato per ognuno degli N input dati alla rete

    
    %Propaghiamo l'input in avanti attraverso ognuno degli strati
    layerInput = input;
    for i=1 : net.hiddenLayersNum + 1 
        A{i} = (layerInput * net.W{i}');
        B = repmat(net.b{i},size(layerInput,1),1);
        A{i} = A{i} + B;
        outputs{i} = net.outputFunctions{i}(A{i});
        %L'output sarà l'input per la prossima propagazione
        layerInput = outputs{i}; 

    end
    
    if softMax
        % Applica il soft-max all'output della rete.
        softmax = exp(outputs{net.hiddenLayersNum + 1}) ./ sum(exp(outputs{net.hiddenLayersNum + 1}), 2);
        outputs{net.hiddenLayersNum + 1} = softmax;
    end
        
        
end

