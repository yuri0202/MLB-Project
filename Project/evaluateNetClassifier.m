function [totalAccuracy] = evaluateNetClassifier(output,target)
% Questa funzione valuta le prestazioni della rete neurale



    % Estraggo le risposte della rete
    netAnswers = zeros(size(output,1),size(output,2));
    
    for i = 1 : size(output,1)
        % Trovo la classe a cui corrisponde il valore di uscita maggiore
        [~, argmax] = max(output(i,:));
        
        % Questa è la risposta della rete
        netAnswers(i,argmax)=1;
        
    end
    
    % Calcolo le risposte della rete contando il numero di valori diversi
    % da 0, ottenuti moltiplicando la matrice dei target con quella delle
    % risposte della rete
    correctAnswers = nnz(netAnswers .* target);
    
    % Calcolo l'accuracy come numero di risposte corrette fratto numero
    % totale degli output
    totalAccuracy = correctAnswers/size(output,1);
        
end

