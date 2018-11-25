function y = sumOfSquares(X,Y)
%Calcola la funzione somma dei quadrati sugli input X e Y.
    
    y = 0.5 * sum(((X - Y) .^ 2));
end

