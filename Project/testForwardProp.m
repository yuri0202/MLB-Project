% Script per testare la forward propagation

% Creo la rete neurale
net = createNeuralNetwork(2, 1, @identity, @identityDx, [
    struct('size',2,'function',@sigmoid,'derivative',@sigmoidDx) % Hidden Layer
],@derivativeSumOfSquares);


disp('Pesi W1');
disp(net.W{1});
disp('Bias liv. 1');
disp(net.b{1});
disp('Pesi W2');
disp(net.W{2});
disp('Bias output');
disp(net.b{2});

% Effettuo forward propagation
[outputs, A] = forwardProp(net, [2,3]);

disp('A strato interno');
disp(A{1});
disp('A output');
disp(A{2});
disp('Z1');
disp(outputs{1});
disp('Y');
disp(outputs{2});

disp('fine');