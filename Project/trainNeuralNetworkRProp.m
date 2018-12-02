function [net,TSErrors, VSErrors] = trainNeuralNetworkRProp(net, trainingSetInput, trainingSetLabels, validationSetInput, validationSetLabels, epochs, errorFunction, errorFunctionDx, etaMinus, etaPlus, softmax, printFlag)
% Questa funzione addestra la rete neurale utilizzando la Resilient Back
% Propagation (RProp)
%
% INPUT:
%   - 'net' rete neurale creata dalla funzione 'createNeuralNetwork'
%   - 'trainingSetInput': Training Set per l'addestramento della rete. La
%                         riga i-esima rappresenta l'i-esimo input per la
%                         rete neurale
%   - 'trainingSetLabels': Matrice che rappresenta il target da ottenere
%                          rispetto ai valori di output generati dalla 
%                          rete neurale con in input il training set
%   - 'validationSetInput': Validation Set utilizzato per evitare
%                           overfitting durante l'addestramento del
%                           training set
%   - 'validationSetLabels': Matrice che rappresenta il target da ottenere
%                            rispetto ai valori di output generati dalla 
%                            rete neurale con in input il validation set
%   - 'epochs': Numero massimo di epoche con cui addestrare la rete
%   - 'errorFunction': Funzione da utilizzare per il calcolo dell'errore
%   - 'errorFunctionDx': Derivata della funzione da utilizzare per il
%                        calcolo dell'errore
%   - 'etaMinus': Numero reale che rappresenta il fattore moltiplicativo
%                 rispetto allo scostamento precedente della matrice dei
%                 pesi e dei bias, quando la derivata della funzione di
%                 errore è discorde con quella precedente
%   - 'etaPlus': Numero reale che rappresenta il fattore moltiplicativo
%                rispetto allo scostamento precedente della matrice dei
%                pesi e dei bias, quando la derivata della funzione di
%                errore è concorde con quella precedente
%   - 'softmax': Se uguale a TRUE, all'output della rete (dopo la forward
%                propagation) verrà applicato il softmax; no altrimenti
%   - 'printFlag': Se uguale a TRUE, verranno stampati a video i valori
%                  degli errori calcolati rispetto al Training e al
%                  Validation Set.
%
% OUTPUT:
%   - 'net': Rete Neurale addestrata sul training set
%   - 'TSErrors': Array di errori tale che l'i-esimo elemento rappresenta
%                 l'errore sul training set alla i-esima epoca
%   - 'VSErrors': Array di errori tale che l'i-esimo elemento rappresenta
%                 l'errore sul validation set alla i-esima epoca


    % Controllo se il numero di nodi dello strato di input è uguale al
    % numero di colonne di trainingSetInput
    if size(trainingSetInput, 2) ~= net.inputDimension
        error("La grandezza del Training Set non è corretta: Il numero di nodi dello strato di input è %d, ma il training ha grandezza %d.",size(trainingSetInput,2), net.inputDimension);
    end
    
    % Controllo se il numero di nodi dello strato di input è uguale al
    % numero di colonne di validationSetInput
    if size(validationSetInput, 2) ~= net.inputDimension
        error("La grandezza del Validation Set non è corretta: Il numero di nodi dello strato di input è %d, ma il validation ha grandezza %d.",size(validationSetInput,2), net.inputDimension);
    end
    
    % Controllo se il numero di nodi dello strato di output è uguale al
    % numero di colonne di trainingSetLabels
    if size(trainingSetLabels, 2) ~= net.outputDimension
        error("La grandezza del TrainingSetLabels non è corretta: Il numero di nodi dello strato di output è %d, ma il TrainingLabel ha grandezza %d.",size(trainingSetLabels,2), net.outputDimension);
    end
    
    % Controllo se il numero di nodi dello strato di output è uguale al
    % numero di colonne di validationSetLabels
    if size(validationSetLabels, 2) ~= net.outputDimension
        error("La grandezza del validationSetLabels non è corretta: Il numero di nodi dello strato di output è %d, ma il validationLabel ha grandezza %d.",size(validationSetLabels,2), net.outputDimension);
    end
    
    
    % Array per gli errori ad ogni epoca
    TSErrors = zeros(1,epochs);
    VSErrors = zeros(1,epochs);
    
    % Conservo la rete migliore e inizializzo parametri per il criterio di
    % stop
    bestNet = net;
    bestVSError = realmax;
    minEpochs = floor(epochs/3);
    errorCount = 0;
    
    % Definisco le strutture per conservare le derivate e i delta di
    % aggiornamento
    derivativeW = cell(1, net.hiddenLayersNum+1);
    derivativeB = cell(1, net.hiddenLayersNum+1);
    deltaW = cell(1, net.hiddenLayersNum+1);
    deltaB = cell(1, net.hiddenLayersNum+1);
    
    % Inizia l'addestramento della rete per ogni epoca
    for epoch = 1 : epochs
        if (printFlag)
            fprintf('EPOCH %d.\n',epoch);
        end
        % Salvo la rete prima di effettuare l'aggiornamento
        prevNet = net;
        
        %Addestramento rete per una epoca -> chiamo funzione
        [net, TSErrors(epoch), VSErrors(epoch), deltaB, deltaW, derivativeB, derivativeW] = resilientBackPropagation(net, trainingSetInput, trainingSetLabels, validationSetInput, validationSetLabels, errorFunction, errorFunctionDx, etaMinus, etaPlus, epoch, derivativeW, derivativeB, deltaW, deltaB, softmax, printFlag);
       
        
        
        % Se l'errore sul validation aumenta per un numero di epoche
        % fissato e sono stati eseguiti un numero minimo di epoche
        % (minEpochs), allora si evita l'overfitting terminando
        % l'addestramento
        
        if VSErrors(epoch) < bestVSError 
            % L'errore sul validation non è aumentato, azzeriamo il
            % contatore e salviamo i valori migliori
            errorCount = 0;
            bestVSError = VSErrors(epoch);
            bestNet = prevNet;
        else % Altrimenti, se è passato un numero minimo di epoche, 
             % incrementiamoil contatore e se abbiamo raggiunto il limite, 
             % fermiamo l'addestramento
             if epoch>=minEpochs
                 errorCount = errorCount+1;
                 if errorCount == 20
                     break;
                 end
             end
        end
    end
    
    % Se abbiamo interrotto il learning con il criterio di stop, riduciamo
    % gli array degli errori
    if epoch < epochs
        TSErrors = TSErrors (1:epoch);
        VSErrors = VSErrors (1:epoch);
    end
    
    % Recupero la rete che ha registrato l'errore minimo sul 
    % Validation Set
    net = bestNet;  
end