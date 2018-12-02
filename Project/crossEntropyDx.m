function y = crossEntropyDx(X,Y)

% La funzione calcola la derivata della Cross Entropy

% La funzione è stata implementata considerando che venisse usata insieme
% all'applicazione del softmax sugli output della rete.

    y = X - Y;

end

