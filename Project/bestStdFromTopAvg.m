function [bestEtaMin,bestEtaPlus, bestNumNodes] = bestStdFromTopAvg(meanStdPerComb, topAvg)
%>> A = [1 2 3 67 1 ; 1 2 1 64 3; 1 2 2 69 1.5; 3 2 1 70 1.2; 1 1 1 80 1.2; 2 2 2 73 1.1; 3 3 3 77 1.25; 3 2 2 81 1.5; 3 2 1 79 1.3; 2 3 1 56 1.1] 

    % Ordino la matrice in input per i valori decrescenti della colonna che 
    % rappresenta la media e prendo i primi 'topAvg' elementi

    A = sortrows(meanStdPerComb,-4);
    B = A(1:topAvg,:);

    % Ordino per i valori crescenti della colonna che rappresenta lo std
    C = sortrows(B,5);

    % Restituisco i valori degli hyper parametri corrispondenti alla
    % combinazione migliore
    bestNumNodes = C(1,1);
    bestEtaMin = C(1,2);
    bestEtaPlus = C(1,3);
end

