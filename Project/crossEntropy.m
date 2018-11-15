function y = crossEntropy(X,Y)
%Calcola la funzione cross entropy sugli input X e Y.
   
    
    tmp = Y;
    % Per i valori maggiori di zero, viene applicata la funzione standard.
    % Per i valori minori o uguali di zero assegniamo un valore negativo
    % molto grande
    tmp(X > 0) = Y(X > 0) .* log(X(X>0));
    tmp(X <= 0) = Y(X <= 0) .* log(realmin('single'));
     y = -sum(tmp);
end

