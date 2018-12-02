function [net,TSE, VSE, deltaB, deltaW, trainingDerivB, trainingDerivW] = resilientBackPropagation(net,trainingSetInput, trainingSetLabels, validationSetInput, validationSetLabels, errorFunction, errorFunctionDx, etaMinus, etaPlus, epoch, derivativeWPrec, derivativeBPrec, deltaWPrec, deltaBPrec,softmax, printFlag)
% Funzione per l'addestramento con la Resilient Back Propagation per una
% singola epoca
%
% INPUT:
%   - 'net' rete neurale creata dalla funzione 'createNeuralNetwork'
%   - 'trainingSetInput': Training Set per l'addestramento della rete. La
%                         riga i-esima rappresenta l'i-esimo input per la
%                         rete neurale
%   - 'trainingSetLabels': Matrice che rappresenta il target da ottenere
%                          rispetto ai valori di output generati dalla 
%                          rete neurale con in input il training set
%                          
%   - 'validationSetInput': Validation Set utilizzato per evitare
%                           overfitting durante l'addestramento del
%                           training set
%   - 'validationSetLabels': Matrice che rappresenta il target da ottenere
%                            rispetto ai valori di output generati dalla 
%                            rete neurale con in input il validation set
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
%   - 'epoch': Epoca corrente
%   - 'derivativeWPrec': Cell Array che associa ad ogni strato di pesi
%                        della rete le derivate della funzione di errore
%                        rispetto ad ogni peso, che son ostate calcolate
%                        nell'epoca precedente.
%   - 'derivativeBPrec': Cell Array che associa ad ogni strato di pesi
%                        della rete le derivate della funzione di errore
%                        rispetto ad ogni bias, che sono state calcolate
%                        nell'epoca precedente.
%   - 'deltaWPrec': Cell array che associa ad ogni strato di pesi, la
%                   variazione effettuata sui pesi nell'epoca precedente
%   - 'deltaBPrec': Cell array che associa ad ogni strato di nodi, la
%                   variazione effettuata su ogni bias nell'epoca 
%                   precedente
%   - 'softmax': Se uguale a TRUE, all'output della rete (dopo la forward
%                propagation) verrà applicato il softmax; no altrimenti
%   - 'printFlag': Se uguale a TRUE, verranno stampati a video i valori
%                  degli errori calcolati rispetto al Training e al
%                  Validation Set.
%
% OUTPUT:
%   - 'net': Rete neurale addestrata sul training set
%   - 'TSE': Errore sul training set all'epoca corrente
%   - 'VSE': Errore sul validation set all'epoca corrente
%   - 'deltaB': Cell array che associa ad ogni strato di nodi la 
%               variazione effettuata per ogni bias nell'epoca corrente
%   - 'deltaW': Cell array che associa ad ogni strato di pesi la 
%               variazione effettuata per ogni peso nell'epoca corrente
%   - 'trainingDerivB': Array cell che associa ad ogni layer di nodi la
%                       derivata della funzione di errore calcolata
%                       rispetto ad ogni bias nell'epoca corrente
%   - 'trainingDerivW': Array cell che associa ad ogni layer di pesi la
%                       derivata della funzione di errore calcolata
%                       rispetto ad ogni peso nell'epoca corrente
    

    % Effettuo la forward propagation per il training e il validation set
    [outputsT, At] = forwardProp(net, trainingSetInput, softmax);
    [outputsV, Av] = forwardProp(net, validationSetInput, softmax);
    
    % Effettuo il calcolo dell'errore totale per entrambi i set
    TSE = sum(errorFunction(outputsT{net.hiddenLayersNum+1}, trainingSetLabels));
    VSE = sum(errorFunction(outputsV{net.hiddenLayersNum+1}, validationSetLabels));
    
    % Se il flag per la stampa è true, stampo i valori della funzione di
    % errore
    if printFlag
        fprintf('Errore sul Training: %f\nErrore sul Validation: %f.\n\n', TSE, VSE);
    end
    
    % Calcolo delle derivate dei pesi e dei bias tramite back propagation
    [trainingDerivW, trainingDerivB] = backProp(net, trainingSetInput, outputsT, At, trainingSetLabels, errorFunctionDx);
    
    % Aggiornamento della rete con Resilient Back Propagation
    
    % Per ogni hidden layer e layer di output della rete, aggiorna pesi e
    % bias con uno scostamento.
    for l = 1 : net.hiddenLayersNum+1
            % Il delta della prima epoca coincide con quello utilizzato
            % nell'algoritmo della discesa del gradiente.
            % Abbiamo scelto empiricamente eta = 0.000001
        if epoch == 1

            deltaW{l} = trainingDerivW{l} * (-0.000001);
            deltaB{l} = trainingDerivB{l} * (-0.000001);
        
        else
            
            % Calcolo la concordanza dei segni tra le derivate di pesi e
            % bias calcolate nell'epoca precedente e quelli calcolati
            % nell'epoca attuale
            concordanceW = derivativeWPrec{l} .* trainingDerivW{l};
            concordanceB = derivativeBPrec{l} .* trainingDerivB{l};
            
            % Inizializzo gli scostamenti
            deltaW{l} = zeros(size(deltaWPrec{l},1),size(deltaWPrec{l},2));
            deltaB{l} = zeros(size(deltaBPrec{l},1),size(deltaBPrec{l},2));
            
        % I pesi e i bias sono aggornati secondo quanto stabilito dalla 
        % Resilient back propagation:
            
            % Se i pesi e i bias hanno derivate concordi, gli scostamenti 
            % dell'epoca precedente vengono incrementati con il valore 
            %'etaPlus'
            deltaW{l}(concordanceW > 0) = deltaWPrec{l}(concordanceW > 0) * etaPlus;
            deltaB{l}(concordanceB > 0) = deltaBPrec{l}(concordanceB > 0) * etaPlus;
            
            
            % Se i pesi e i bias hanno derivate discordi, gli scostamenti
            % dell'epoca precedente vengono decrementati per il valore
            % 'etaMinus' cambiato di segno
            deltaW{l}(concordanceW < 0) = deltaWPrec{l}(concordanceW < 0) * (-etaMinus);
            deltaB{l}(concordanceB < 0) = deltaBPrec{l}(concordanceB < 0) * (-etaMinus);
            
            % Se la moltiplicazione tra derivata precedente e derivata
            % corrente è pari a 0, allora il valore del nuovo delta è
            % uguale a quello precedente
            deltaW{l}(concordanceW==0) = deltaWPrec{l}(concordanceW==0);
            deltaB{l}(concordanceB==0) = deltaBPrec{l}(concordanceB==0);
        end
        
        % Aggiorno pesi e bias usando gli scostamenti
        net.b{l} = net.b{l} + deltaB{l};
        net.W{l} = net.W{l} + deltaW{l};
    end
end
    
    


