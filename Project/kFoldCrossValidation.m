function [Performances] = kFoldCrossValidation(trainingSetData, trainingSetLabels, trainingSetSize, foldSize,meanStdCurrentIndex, numIterations, K, HIDDEN_NODES, ETA_MINUS, ETA_PLUS, epochs, outputFunction, outputFunctionDx,hiddenFunction,hiddenFunctionDx,supWeights,infWeights,errorFunction,errorFuctionDx,softMax,printFlag);
% Questa funzione effettua la K-Fold Cross Validation su una singola
% combinazione di iper-parametri
% INPUT:
%   - 'trainingSetData': Set sul quale effettuare la K-Fold Cross
%                        Validation
%   - 'trainingSetLabels': target da ottenere rispetto alle immagini di
%                          'trainingSetData'
%   - 'trainingSetSize': Numero di elementi presenti in 'trainingSetData'
%   - 'foldSize': Numero di elementi per ogni fold (trainingSetSize/K)
%   - 'meanStdCurrentIndex': Indice che tiene traccia dell'ultimo elemento
%                            inserito nella matrice meanStdCurrentIndex per
%                            il calcolo della media e della deviazione
%                            standard delle performances
% - 'numIterations': Numero totale di iterazioni da effettuare, dato dalla
%                    moltiplicazione delle cardinalità dell'array di
%                    possibili valori per etaMinus, etaPlus e Numero di
%                    nodi dello strato interno
%   - 'K': Numero di Fold con cui effettuare la K-Fold
%   - 'HIDDEN_NODES': Il numero di nodi dello strato interno per la
%                     combinazione corrente di iper-parametri
%   - 'ETA_MINUS': Il valore di etaMinus per la
%                  combinazione corrente di iper-parametri
%   - 'ETA_PLUS': Il valore di etaPlus per la
%                  combinazione corrente di iper-parametri
%   - 'epochs': Numero massimo di epoche con cui addestrare la rete

%   - 'outputFunction': Funzione di attivazione dello strato di output
%   - 'outputDerivative': Derivata della funzione di attivazione dello
%                         strato di output
%   - 'hiddenFunction': Funzione di attivazione dello strato interno
%   - 'hiddenFunctionDx': Derivata della funzione di attivazione dello
%                         strato interno
%   - 'supWeights': Estremo superiore dell'intervallo dei valori con cui
%                   riempire casualmente i pesi della rete.
%   - 'infWeights': Estremo inferiore dell'intervallo dei valori con cui
%                   riempire casualmente i pesi della rete.
%   - 'errorFunction': Funzione da utilizzare per il calcolo dell'errore
%   - 'errorFunctionDx': Derivata della funzione da utilizzare per il
%                        calcolo dell'errore
%   - 'softmax': Se uguale a TRUE, all'output della rete (dopo la forward
%                propagation) verrà applicato il softmax; no altrimenti
%   - 'printFlag': Se uguale a TRUE, verranno stampati a video i valori
%                  degli errori calcolati rispetto al Training e al
%                  Validation Set durante la fase di RProp

    % Vengono definiti gli indici iniziali per la prima
    % partizione del set in Ts, Vs e Tes
    TsStart = 1;
    TsEnd = (K-2)*foldSize;
    VsStart = (K-2)*foldSize+1;
    VsEnd = (K-1)*foldSize;
    TesStart = (K-1)*foldSize +1;
    TesEnd = trainingSetSize;

    % Array dove salvo le performances per ogni iterazione del
    % K-Fold 
    Performances= zeros(1,K);
    
    fprintf("\nParametri del K-fold corrente: numero nodi = %d, etaMinus =%.2f, etaPlus = %.2f\n", HIDDEN_NODES,ETA_MINUS,ETA_PLUS);
    fprintf("Iterazione %d di %d\n\n",meanStdCurrentIndex,numIterations);
    for k = 1 : K
        % Split in TS, Vs e TeS in accordo con gli indici calcolati
        % all'iterazione precedente

        if TsStart > TsEnd
            % Se l'indice di fine set del Training Set è maggiore dell'indice
            % di inizio set (Esempio[K=10] Training Set = folds 4 5 6 7
            % 7 9 10 1 2), allora il TS sarà ovviamente composto dagli
            % elementi che vanno dal fold 4 al fold K e quelli dal
            % fold 1 al fold 2
            TSData = trainingSetData([TsStart:trainingSetSize,1:TsEnd],:);
            TSLabels = trainingSetLabels([TsStart:trainingSetSize,1:TsEnd],:);
        else
            TSData = trainingSetData(TsStart:TsEnd,:);
            TSLabels = trainingSetLabels(TsStart:TsEnd,:);
        end
        VSData = trainingSetData(VsStart:VsEnd,:);
        VSLabels = trainingSetLabels(VsStart:VsEnd,:);
        TeSData = trainingSetData(TesStart:TesEnd,:);
        TeSLabels = trainingSetLabels(TesStart:TesEnd,:);


        % Creo la rete neurale con numero di nodi dell'hidden layer pari a
        % HIDDEN_NODES

        net = createNeuralNetwork(size(trainingSetData,2), size(trainingSetLabels,2), outputFunction, outputFunctionDx, [
            struct('size',HIDDEN_NODES,'function',hiddenFunction,'derivative',hiddenFunctionDx) % Hidden Layer
            ], supWeights,infWeights);

        % Addestro la rete con la combinazione di parametri di
        % questa iterazione su Training e Validation
        [net,~, ~] = trainNeuralNetworkRProp(net, TSData, TSLabels, VSData, VSLabels, epochs, errorFunction,errorFuctionDx, ETA_MINUS, ETA_PLUS, softMax, printFlag);

        % Forward propagation della rete addestrata utilizzando come input
        % il test set di questa iterazione di K-Fold
        [output, ~] = forwardProp(net,TeSData, true);
        
        % Calcolo dell'accuratezza delle risposte della rete, confrontandole con le
        % label effettive del test set.
        [totalAccuracy] = evaluateNetClassifier(output{net.hiddenLayersNum+1}, TeSLabels);
        totalAccuracy = totalAccuracy*100;
        Performances(k) = totalAccuracy;
        fprintf("Accuracy della rete neurale - Itarazione kfold %d di %d: %.2f%%\n", k,K,totalAccuracy);

        % Aggiorno gli indici per la prossima iterazione
        % Utilizzo la funzione realizzata myMod tale che dati
        % due numeri a e b, restituisce il mod(a,b) solo se a > b.
        % Sostanzialmente è uguale alla funzione mod tranne per
        % il fatto che il mod(x,x) = x e non 0
        TsStart = myMod(TsStart+foldSize,trainingSetSize);
        TsEnd = myMod(TsEnd+foldSize,trainingSetSize);
        VsStart = myMod(VsStart+foldSize,trainingSetSize);
        VsEnd = myMod(VsEnd+foldSize,trainingSetSize);
        TesStart = myMod(TesStart+foldSize,trainingSetSize);
        TesEnd = myMod(TesEnd+foldSize,trainingSetSize);
    end
end


function [myMod] = myMod (number,module)
% Questa funzione, utilizzata durante l'aggiornamento degli indici per i
% vari set tra una iterazione e l'altra della K-Fold Cross Validation, è
% una modifica alla classica funzione di modulo. Si differenzia dal fatto
% che myMod(x,x) = x e non 0 come nella classica funzione di mod.
    
    if (number == module)
        myMod = number;
    else
        myMod = mod(number,module);
    end
end


