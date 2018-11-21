function[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = createSets(trainImages, trainLabels, testImages, testLabels, trainingSetSize, validationSetSize, testSetSize)
% Questa funzione estrae casualmente dal dataset MNIST una matrice per il
% training, una per il validation set e una per il test set.
% Training e Validation set saranno estratti dal file del dataset MNIST
% relativo al training set, mentre il test set sarà estratto dal file
% relativo al test set.
% Ognuno dei set contiene valori distinti e la sua dimensione dipende dai parametri
% forniti in input. Al fine di avere un numero di immagini uguale per ogni
% classe, le grandezze dei vari set dati in input dovranno essere
% interi multipli di 10, altrimenti la funzione lancerà un errore.

% INPUT:
%   - 'trainImages': Matrice di immagini dal training set di MNIST
%                    (60000x784), ottenuta dalla funzione 'loadMNIST'
%   - 'trainLabels': Matrice di labels dal training set di MNIST (60000x1),
%                    ottenuta dalla funzione 'loadMNIST' 
%   - 'testImages': Matrice di immagini dal testset di MNIST
%                    (10000x784), ottenuta dalla funzione 'loadMNIST'
%   - 'testLabels': Matrice di labels dal test set di MNIST (10000x1),
%                    ottenuta dalla funzione 'loadMNIST' 
%   - 'trainingSetSize': Numero di dati da estrarre per il training set.
%                        Deve essere un numero multiplo di 10
%   - 'validationSetSize': Numero di dati da estrarre per il validation set.
%                        Deve essere un numero multiplo di 10
%   - 'testSetSize: Numero di dati da estrarre per il test set.
%                       Deve essere un numero multiplo di 10
%
% OUTPUT:
%   - 'trainingSetData': Matrice di dimensione [trainingSetSize]x784 che
%                        contiene l'insieme di immagini distinte e casuali
%                        che sono state estratte da trainImages
%   - 'trainingSetlabels': Matrice di dimensione [trainingSetSize]x10 che
%                          rappresenta le labels delle immagini in
%                          'trainingSetData'. (La i-esima riga avrà tutti 0
%                          tranne un 1 in corrispondenza della label
%                          relativa all'immagine i-esima di
%                          trainingSetData)
%   - 'validationSetData': Matrice di dimensione [validationSetSize]x784 che
%                          contiene l'insieme di immagini distinte e casuali
%                          che sono state estratte da trainImages
%   - 'validationSetlabels': Matrice di dimensione [validationSetSize]x10 che
%                            rappresenta le labels delle immagini in
%                            'validationSetData'. (La i-esima riga avrà 
%                            tutti 0 tranne un 1 in corrispondenza della
%                            label relativa all'immagine i-esima di
%                            'validationSetData')
%   - 'testSetData': Matrice di dimensione [testetSize]x784 che
%                    contiene l'insieme di immagini distinte e casuali
%                    che sono state estratte da testImages
%   - 'testSetlabels': Matrice di dimensione [testSetSize]x10 che
%                      rappresenta le labels delle immagini in
%                      'testSetData'. (La i-esima riga avrà tutti 0 tranne
%                      un 1 in corrispondenza della label relativa
%                      all'immagine i-esima di 'testSetData')


    % Controllo sulle dimensioni dei set dati in input, se una dimensione
    % non è un multiplo di 10, allora la funzione termina con un errore
    if (mod(trainingSetSize, 10) ~= 0) || (mod(validationSetSize, 10) ~= 0) || (mod(testSetSize, 10) ~= 0)
        error('Il numero di elementi da selezionare per ogni set deve essere un multiplo di 10');
        return
    end
    
    % Creo un array per tenere traccia degli indici che sono stati già
    % selezionati da trainImages per garantire di avere set disgiunti e
    % distinti per training e validation set
    indexesTaken = zeros(1, (trainingSetSize+validationSetSize));
    % Conservo l'ultima posizione di indexesTaken dove è stato inserito un
    % valore
    lastPosition = 1;
    
    % Definisco il numero totale di elementi presenti in trainImages, come
    % limite massimo per la scelta randomica di indice
    totSize = 60000;
    
    % Calcolo la matrice delle immagini e delle labels per il training set.
    % Ritorno l'array indexesTaken aggiornato con gli indici delle immagini
    % inserite nel training set
    [trainingSetData, trainingSetLabels, indexesTaken, lastPosition] = buildSet(trainImages, trainLabels, trainingSetSize, indexesTaken, lastPosition, totSize);
    
    % Calcolo la matrice delle immagini e delle labels per il validation set.
    [validationSetData, validationSetLabels] = buildSet(trainImages, trainLabels, validationSetSize, indexesTaken, lastPosition, totSize);
    
    % Ridefinisco l'array indexesTaken per tenere traccia degli indici già
    % selezionati da testImages, per garantire di avere tutti elementi
    % distinti all'interno del testSet
    indexesTaken = zeros(1, (testSetSize));
    
    % Conservo l'ultima posizione di indexesTaken dove è stato inserito un
    % valore
    lastPosition = 1;
    
        
    % Definisco il numero totale di elementi presenti in testImages, come
    % limite massimo per la scelta randomica di indice
    totSize = 10000;
    
    [testSetData, testSetLabels] = buildSet(testImages,testLabels,testSetSize, indexesTaken, lastPosition, totSize);
end


function [setData, setLabels, indexesTaken, lastPosition] = buildSet(digits, labels, setSize, indexesTaken, lastPosition, totSize)

    % Contatore per incrementare gli indici della matrice delle immagini e
    % delle labels da estrarre dal dataset
    j = 1;
    % Matrice delle immagini
    setData = zeros(setSize, 784);
    % Matrice delle labels
    setLabels = zeros(setSize, 10);
    % Numero di immagini da estrarre per ogni cifra
    elemsForDigits = setSize/10;
    % Array che tiene traccia del numero di immagini inserite per ogni
    % cifra. La cella i-esima corrisponde al valore della cifra i-esima
    digitsCounter = zeros(1, 10);
    currentDigit = 0;
    
    % Loop su tutte le cifre (0-9)
    while currentDigit <= 9
        % Fino a che non ho estratto un numero pari a 'elemsForDigits' per
        % ogni cifra:
        while digitsCounter(currentDigit+1) <= elemsForDigits-1
            % Genera un numero casuale da 1 a totSize (600000 o 10000)
            randomIndex = floor((totSize-1).*rand(1) + 1);
            % Se il numero generato corrisponde all'indice di un'immagine
            % che rappresenta la cifra corrente, e se questa immagine non è
            % già stata inserita in 'setData' allora posso aggiungerla
            if (labels(randomIndex) == currentDigit) && (~ismember(randomIndex, indexesTaken))
                % Incremento il numero di immagini aggiunte per la cira corrente
                digitsCounter(currentDigit+1) = digitsCounter(currentDigit+1) + 1;
                % Aggiungo la label e l'immagine alle matrici di output
                setLabels(j, currentDigit+1) = 1;
                setData(j, :) = digits(randomIndex, :);
                
                % Aggiungi questo indice a quelli gia' inseriti
                indexesTaken(lastPosition) = randomIndex;
                j = j + 1;
                lastPosition = lastPosition + 1;
            end
        end
        currentDigit = currentDigit + 1;
    end
end