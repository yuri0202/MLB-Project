function [etaMinusBest,etaPlusBest, numHiddenNodesBest] = kFoldCrossValidation(trainingSetSize,epochs,outputFunction,outputFunctionDx,hiddenFunction,hiddenFunctionDx,supWeights,infWeights,errorFunction,errorFuctionDx,softMax,K,etaMins,etaPlus,numHiddenNodes, printFlag)
% poi si vedrà

    % Controlla che trainingSetSize sia multiplo di K 
    if (mod(trainingSetSize, K) ~= 0) 
        error('Il numero di elementi su cui effettuare la K-fold cross validation deve essere multiplo di K');
        return
    end

    % Carico il training set del dataset MNIST
    [trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
   
    
    [trainingSetData, trainingSetLabels, ~, ~, ~, ~] = createSets(trainImages', trainLabels, trainImages', trainLabels, trainingSetSize, 0, 0);
    %trainingSetData = [1 1 1 1 1 1 1 1 1 1; 2 2 2 2 2 2 2 2 2 2; 3 3 3 3 3 3 3 3 3 3; 4 4 4 4 4 4 4 4 4 4; 5 5 5 5 5 5 5 5 5 5; 6 6 6 6 6 6 6 6 6 6; 7 7 7 7 7 7 7 7 7 7; 8 8 8 8 8 8 8 8 8 8; 9 9 9 9 9 9 9 9 9 9; 10 10 10 10 10 10 10 10 10 10];
    foldSize = trainingSetSize/K;
    
    % PROBABILMENTE BISOGNA SHUFFLARE LE RIGHE DELLA MATRICE O SI AVRANNO
    % RISULTATI MEH perchè la funzione createSets restituisce i valori in ordine di
    % cifra
    
    shuffle   = randperm(trainingSetSize);
    trainingSetData   = trainingSetData(shuffle,:);
    trainingSetLabels = trainingSetLabels(shuffle,:);
    

    
    % Matrice che conserva per ogni combinazione di valori degli
    % iper-parametri, la media delle performance ottenute tramite k-fold
    % cross validation
    avgPerComb = zeros(size(etaMins,1)*size(etaPlus,1)*size(numHiddenNodes,1),4);
    
    % Inizio le iterazioni sui possibili valori per gli iper-parametri
    for i=1 : size(numHiddenNodes,2)     
        for j = 1 : size(etaMins,2)
            for z = 1 : size(etaPlus,2)
                % Assegno gli iper-parametri di questa iterazione
                ETA_MINUS = etaMins(j);
                ETA_PLUS = etaPlus(z);
                
                % Probabilmente questa parte qui metterla in una sola
                % funzione che chiamiamo kFoldCrossValidation mentre questa
                % funzione qui la chiamiamo tuneHyperParametersCV o qualcosa
                % del genere
                
                % Vengono definiti gli indici iniziali per la prima
                % partizione del set in Ts, Vs e Tes
                TsStart = 1;
                TsEnd = (K-2)*foldSize;
                VsStart = (K-2)*foldSize+1;
                VsEnd = (K-1)*foldSize;
                TesStart = (K-1)*foldSize +1;
                TesEnd = trainingSetSize;
                
                % Array dove salvo le performances per ogni iterazione del
                % k-fold (se facciamo funzione interna questo array verrà
                % ritornato dalla funzione)
                Performances= zeros(1,K);
                for k = 1 : K
                    % Se il Training Set inizia in un fold < K e finisce in
                    % un fold > K (Esempio[K=10] Training Set = folds 4 5 6 7
                    % 7 9 10 1 2), allora sarà ovviamente composto dagli
                    % elementi che vanno dal fold 4 al fold K e quelli dal
                    % fold 1 al fold 2
                    if TsStart > TsEnd
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
                    % numHiddenNodes(i)
                    HIDDEN_NODES = numHiddenNodes(i);
                    net = createNeuralNetwork(size(trainingSetData,2), size(trainingSetLabels,2), outputFunction, outputFunctionDx, [
                        struct('size',HIDDEN_NODES,'function',hiddenFunction,'derivative',hiddenFunctionDx) % Hidden Layer
                    ], supWeights,infWeights);
                    
                    % Addestro la rete con la combinazione di parametri di
                    % questa iterazione su Training e Validation
                    [net, trainingSetErrors, validationSetErrors] = trainNeuralNetworkRProp(net, TSData, TSLabels, VSData, VSLabels, epochs, errorFunction,errorFuctionDx, ETA_MINUS, ETA_PLUS, softMax, printFlag);
                    
                    % Calcolo l'accuracy
                    [output, ~] = forwardProp(net,TeSData, true);
                    [totalAccuracy] = evaluateNetClassifier(output{net.hiddenLayersNum+1}, TeSLabels);
                    totalAccuracy = totalAccuracy*100;
                    Performances(k) = totalAccuracy;
                    fprintf("\n Accuracy della rete neurale, Itarazione kfold %d: %.2f%% \n", k,totalAccuracy);
                    
                    % 5) Calcolo la media e deviazione standard salvo la media dentro
                    %    avgPerComb con i 3 parametri di riferimento
                    % 6) Se la media è minore della media migliore fino ad
                    % ora, aggiorna il valore dei parametri migliori
                    
                    % Aggiorno gli indici per la prossima iterazione
                    % Utilizzo la funzione realizzata myMod tale che dati
                    % due numeri a e b, restituisce il mod(a,b) se a > b.
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
        end
    end
    
end


function [myMod] = myMod (number,module)
    
    if (number == module)
        myMod = number;
    else
        myMod = mod(number,module);
    end
end


