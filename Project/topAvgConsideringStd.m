function [bestNumNodes, bestEtaMin,bestEtaPlus ] = topAvgConsideringStd(meanStdPerComb, stdThreshold)
   

    % Ordino la matrice in input per i valori decrescenti della colonna che 
    % rappresenta la media e prendo i primi 'topAvg' elementi

    A = sortrows(meanStdPerComb,-4);
    
    numRows = size(A,1);
    numCols = size(A,2);
    
    returnIndex = 1;
    for i = 1 : numRows
        if A(i,numCols) <= stdThreshold
            returnIndex = i;
            break;          
        end     
    end
    
    

    % Restituisco i valori degli hyper parametri corrispondenti alla
    % combinazione migliore
    bestNumNodes = A(returnIndex,1);
    bestEtaMin = A(returnIndex,2);
    bestEtaPlus = A(returnIndex,3);
end

