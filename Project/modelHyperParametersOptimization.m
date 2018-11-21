function [meanStdPerComb] = modelHyperParametersOptimization(trainImages,trainLabels,trainingSetSize,epochs,outputFunction,outputFunctionDx,hiddenFunction,hiddenFunctionDx,supWeights,infWeights,errorFunction,errorFuctionDx,softMax,K,etaMins,etaPlus,numHiddenNodes, printFlag)
% Questa funzione trova la combinazione di parametri ottimi (etaMinus,
% etaPlus e numero di nodi dello strato interno) per l'addestramento 
% della rete, sfruttando la tecnica del K-Fold Cross Validation

%
% INPUT:
%   - 'trainImages': Matrice di immagini dal training set di MNIST
%                    (60000x784), ottenuta dalla funzione 'loadMNIST'
%   - 'trainLabels': Matrice di labels dal training set di MNIST (60000x1),
%                    ottenuta dalla funzione 'loadMNIST' 
%   - 'trainingSetSize': Numero di elementi da prendere da trainImages sul
%                        quale effettuare la K-Fold
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
%   - 'K': Numero di Fold con cui effettuare la K-Fold
%   - 'etaMins': Array che contiene i diversi valori di etaMin con le quali
%                provare le diverse combinazioni
%   - 'etaPlus': Array che contiene i diversi valori di etaPlus con le quali 
%                provare le diverse combinazioni
%   - 'numHiddenNodes': Array che contiene i diversi valori di numero di
%                       nodi interni con i quali provare le diverse
%                       combinazioni
%   - 'printFlag': Se uguale a TRUE, verranno stampati a video i valori
%                  degli errori calcolati rispetto al Training e al
%                  Validation Set durante la fase di RProp

% OUTPUT:
%   - 'meanStdPerComb': Matrice che, per ogni combinazione di parametri,
%                       conserva la media e la deviazione standard delle
%                       performances ottenute nelle K iterazioni del
%                       processo di K-Fold Cross Validation

    % Controlla che trainingSetSize sia multiplo di K 
    if (mod(trainingSetSize, K) ~= 0) 
        error('Il numero di elementi su cui effettuare la K-fold cross validation deve essere multiplo di K');
        return
    end
   
    
    % Estraggo elementi random e distinti dal training set MNIST
    [trainingSetData, trainingSetLabels, ~, ~, ~, ~] = createSets(trainImages', trainLabels, trainImages', trainLabels, trainingSetSize, 0, 0);
    %trainingSetData = [1 1 1 1 1 1 1 1 1 1; 2 2 2 2 2 2 2 2 2 2; 3 3 3 3 3 3 3 3 3 3; 4 4 4 4 4 4 4 4 4 4; 5 5 5 5 5 5 5 5 5 5; 6 6 6 6 6 6 6 6 6 6; 7 7 7 7 7 7 7 7 7 7; 8 8 8 8 8 8 8 8 8 8; 9 9 9 9 9 9 9 9 9 9; 10 10 10 10 10 10 10 10 10 10];
    %trainingSetData = [1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 2 2 2 2 2 2 2 2 2 2; 2 2 2 2 2 2 2 2 2 2; 2 2 2 2 2 2 2 2 2 2; 3 3 3 3 3 3 3 3 3 3; 3 3 3 3 3 3 3 3 3 3; 3 3 3 3 3 3 3 3 3 3];

    foldSize = trainingSetSize/K;
    
    % TODO
    % PROBABILMENTE BISOGNA SHUFFLARE LE RIGHE DELLA MATRICE O SI AVRANNO
    % RISULTATI MEH perchè la funzione createSets restituisce i valori in ordine di
    % cifra (Alternativa è creare una funzione che prende in input
    % trainingSetData e trainingSetLabels e bilancia il tutto (es. invece di
    % avere 111,222,333 si avrà 123,123,123)
    
    % Effettuo una permutazione random degli elementi di trainingSetData e
    % trainingSetLabels, poichè restituiti in ordine di digits
    %shuffle   = randperm(trainingSetSize);
    %trainingSetData   = trainingSetData(shuffle,:);
    %trainingSetLabels = trainingSetLabels(shuffle,:);
    
    [trainingSetData, trainingSetLabels] = balanceDataSets(trainingSetData, trainingSetLabels);
    

    
    % Definisco la matrice che conserva per ogni combinazione di valori degli
    % iper-parametri, la media e la deviazione standard delle performance 
    % ottenute tramite k-fold cross validation e l'indice che mi tiene 
    % traccia dell'ultimo elemento inserito
    numIterations = size(etaMins,2)*size(etaPlus,2)*size(numHiddenNodes,2);
    meanStdPerComb = zeros(numIterations,5);
    meanStdCurrentIndex = 1;
    
   
    % Inizio le iterazioni sui possibili valori per gli iper-parametri
    for i=1 : size(numHiddenNodes,2)    
        HIDDEN_NODES = numHiddenNodes(i);
        for j = 1 : size(etaMins,2)
            ETA_MINUS = etaMins(j);
            for z = 1 : size(etaPlus,2)
                ETA_PLUS = etaPlus(z);
                
                % Eseguo la K-Fold Cross Validation con la combinazione di
                % parametri corrente e ne salvo le performances per ogni
                % K-iterazione
                Performances = kFoldCrossValidation(trainingSetData, trainingSetLabels, trainingSetSize, foldSize, meanStdCurrentIndex, numIterations, K, HIDDEN_NODES, ETA_MINUS, ETA_PLUS, epochs, outputFunction, outputFunctionDx,hiddenFunction,hiddenFunctionDx,supWeights,infWeights,errorFunction,errorFuctionDx,softMax,printFlag);

                % Aggiorno la matrice meanStdPerComb con il valore medio e
                % la deviazione standard di tutte le performances calcolate
                % nelle K iterazioni
                meanStdPerComb(meanStdCurrentIndex,1) = HIDDEN_NODES;
                meanStdPerComb(meanStdCurrentIndex,2) = ETA_MINUS;
                meanStdPerComb(meanStdCurrentIndex,3) = ETA_PLUS;
                meanStdPerComb(meanStdCurrentIndex,4) = mean(Performances);
                meanStdPerComb(meanStdCurrentIndex,5) = std(Performances,1); % Divido per N e non uso la correzione di Bessel
                meanStdCurrentIndex = meanStdCurrentIndex +1;
            end
        end
    end  
end


function [trainingSetDataBalanced, trainingSetLabelsBalanced] = balanceDataSets (trainingSetData,trainingSetLabels)
    digits = size(trainingSetLabels,2);
    digitsOcc = size(trainingSetData,1) / digits;
    trainingSetDataBalanced = zeros(size(trainingSetData,1),size(trainingSetData,2));
    j = 1;
    i = 1;
    while i < digitsOcc + 1
        t = i;
        while t < size(trainingSetData,1) + 1
            trainingSetDataBalanced(j,:) = trainingSetData(t,:);
            trainingSetLabelsBalanced(j,:) = trainingSetLabels(t,:);
            t = t + digitsOcc;
            j = j + 1;
        end
     i = i+1;
    end
   
end



