function [y] = sigmoidDx( x )
% Calcola la derivata della funzione sigmoide sull'input x
    y = sigmoid(x) .* sigmoid(1-x); 
end

