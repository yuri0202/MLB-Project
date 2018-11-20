function [etaMinusBest,etaPlusBest, numHiddenNodesBest] = kFoldCrossValidation(net,trainingSetSize,epochs,errorFunction,errorFuctionDx,softmax,K,etaMins,etaPlus,numHiddenNodes)
% poi si vedrà
    
    % Controlla che traiingSetSize sia multiplo di K 
    
    
    % Carico il training set del dataset MNIST
    [trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
   
    
    %[trainingSetData, trainingSetLabels, ~, ~, ~, ~] = createSets(trainImages', trainLabels, trainImages', traintLabels, trainingSetSIze, 0, 0);
    trainingSetData = [1 1 1 1 1 1 1 1 1 1; 2 2 2 2 2 2 2 2 2 2; 3 3 3 3 3 3 3 3 3 3; 4 4 4 4 4 4 4 4 4 4; 5 5 5 5 5 5 5 5 5 5; 6 6 6 6 6 6 6 6 6 6; 7 7 7 7 7 7 7 7 7 7; 8 8 8 8 8 8 8 8 8 8; 9 9 9 9 9 9 9 9 9 9; 10 10 10 10 10 10 10 10 10 10];
    slice = trainingSetSize/K;
    
    % PROBABILMENTE BISOGNA SHUFFLARE LE RIGHE DELLA MATRICE O SI AVRANNO
    % RISULTATI MEH perchè la funzione restituisce i valori in ordine di
    % cifra
    
    % shuffle   = randperm(trainingSetSize);
    % trainingSetData   = trainingSetData(shuffle,:);
    % trainingSetLabels = trainingSetLabels(shuffle,;);
    


    
    % Matrice che conserva per ogni combinazione di valori degli
    % iper-parametri, la media delle performance ottenute con il k-fold
    avgPerComb = zeros(size(etaMins,1)*size(etaPlus,1)*size(numHiddenNodes,1),4);
    
    for i=1 : size(etaMins)
        for j = 1 : size(etaPlus)
            for z = 1 : size(numHiddenNodes)
                % definisci gli indici per ts, vs e tes
                
                % Array dove salvo le performances per ogni iterazione del
                % k-fold
                Performances = zeros(1,K);
                for k = 1 : K
                    TS = trainingSetData(startTs:endTs,:);
                    VS = trainingSetData(startVs:endVs,:);
                    TeS = trainingSetData(startTeS:endTeS,:);
                    
                   
                    % 1) Traina net con Rprop su ts e vs
                    % 2) Testa con forward prop i risultati su Tes
                    % 3) Salva quello che decidiamo di salvare (Accuracy?
                    %    Precision? F-Measure?) dentro Performances
                    % 4) Calcolo la media e salvo la media dentro
                    %    avgPerComb con i 3 parametri di riferimento
                    % 5) Ridefinisci gli indici ruotando
                    % 6) Se la media è minore della media migliore fino ad
                    % ora, aggiorna il valore dei parametri migliori
                    

      
                end                
            end
        end
    end
    
end

