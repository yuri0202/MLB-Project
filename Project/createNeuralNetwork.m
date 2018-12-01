function [ neuralNetwork ] = createNeuralNetwork( inputDimension, outputDimension, outputFunction, outputDerivative, hiddenLayers,infWeights, supWeights )
% La funzione crea una rete neurale feed-forward multistrato
%
%   INPUT:
%       - 'inputDimension': Numero di nodi dello strato di input
%       - 'outputDimension': Numero di nodi dello strato di output
%       - 'outputFunction': Funzione di attivazione dello strato di output
%       - 'outputDerivative': Derivata della funzione di attivazione dello
%                             strato di output
%       - 'hiddenLayers': Array che rappresenta l'insieme degli hidden
%                         layers. Ognuno di essi è così composto:
%                         - 'size': numero di nodi
%                         - outputFunction: Funzione di attivazione
%                         - derivative: La derivata della funzione di
%                         attivazione
%       - 'infWeights': Estremo inferiore dell'intervallo dei valori con cui
%                       riempire casualmente i pesi della rete.
%       - 'supWeights': Estremo superiore dell'intervallo dei valori con cui
%                       riempire casualmente i pesi della rete.
%
%   OUTPUT:
%       - 'neuralNetwork': Rete Neurale costituita dai seguenti campi:
%           - 'inputDimension': Numero di nodi dello strato di input
%           - 'outputDimension': Numero di nodi dello strato di output
%           - 'hiddenLayersNum': Numero di hidden layers
%           - 'n': Array tale per cui l'elemento i-esimo rappresenta il
%                  numero di nodi presenti nel layer i
%           - 'W': Array di matrici bidimensionali dove l'i-esimo elemento
%                  rappresenta i pesi sulle connessioni tra il layer i e il
%                  layer i-1 della rete
%           - 'b': Array di matrici monodimensionali dove l'i-esimo
%                  elemento rappresenta i valori dei bias del layer i
%           - 'outputFunctions': Array tale per cui l'i-esimo elemento
%                                rappresenta la funzione di attivazione del
%                                layer i
%                                
%           - 'derivativeFunctions ': Array tale per cui l'i-esimo elemento
%                                     rappresenta la derivata della
%                                     funzione di attivazione del layer i


    % Controllo se gli elementi dell'array hiddenLayers sono rappresentati
    % per riga o per colonna. Nel primo caso, effettuo la trasposizione
    % dell'array per rispettare la notazione che prevede gli elementi
    % rappresentati per colonne (per coerenza futura)
    if size(hiddenLayers,1) > size(hiddenLayers,2)
        hiddenLayers = hiddenLayers';
    end
    
    neuralNetwork.inputDimension  = inputDimension;
    neuralNetwork.outputDimension = outputDimension;
    neuralNetwork.hiddenLayersNum = size(hiddenLayers,2);
    
    
    % Riempio l'array n con il numero di nodi di ogni layer
    for l = 1 : neuralNetwork.hiddenLayersNum
        neuralNetwork.n(l) = hiddenLayers(l).size;
    end
    
    % Generazione di pesi e bias tra lo strato di input e il primo hidden
    % layer e assegnazione delle funzioni di attivazione e della sua derivata
    neuralNetwork.W{1} = (supWeights-infWeights) .* rand(hiddenLayers(1).size, inputDimension) +infWeights;
    neuralNetwork.b{1} = (supWeights-infWeights) .* rand(1, hiddenLayers(1).size) +infWeights;
    neuralNetwork.outputFunctions{1} = hiddenLayers(1).function;
    neuralNetwork.derivativeFunctions{1} = hiddenLayers(1).derivative;
    
    % Se vi e' piu' di un hidden layer, allora generare pesi e bias per le
    % connessioni tra questi ultimi e assegnare le funzioni di attivazione
    % e la derivata per ogni hidden layer.
    if(size(hiddenLayers,2)>1)
        for i = 2 : size(hiddenLayers,2)
            neuralNetwork.W{i} = (supWeights-infWeights) .* rand(hiddenLayers(i).size,hiddenLayers(i-1).size) + infWeights;
            neuralNetwork.b{i} = (supWeights-infWeights) .* rand(1, hiddenLayers(i).size) + infWeights;
            neuralNetwork.outputFunctions{i} = hiddenLayers(i).function;
            neuralNetwork.derivativeFunctions{i} = hiddenLayers(i).derivative;
        end
    end
    
    % Generazione di pesi e bias tra l'ultimo hidden layer e lo strato di
    % output e assegnazione delle funzioni di attivazione e della sua
    % derivata
    i = length(hiddenLayers) + 1;   
    neuralNetwork.W{i} = (supWeights-infWeights) .* rand(outputDimension, hiddenLayers(i-1).size) + infWeights;
    neuralNetwork.b{i} = (supWeights-infWeights) .* rand(1, outputDimension) + infWeights;
    neuralNetwork.outputFunctions{i} = outputFunction;
    neuralNetwork.derivativeFunctions{i} = outputDerivative;
    
end

