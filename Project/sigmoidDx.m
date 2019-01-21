function [y] = sigmoidDx( x )
% Calcola la derivata della funzione sigmoide sull'input x
    y = sigmoid(x) .* (1 - sigmoid(x)); 
end

