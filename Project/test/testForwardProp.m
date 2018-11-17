% Script per testare la forward propagation

% Creo la rete neurale
net = createNeuralNetwork(5, 1, @sigmoid, @sigmoidDx, [
    struct('size',5,'function',@sigmoid,'derivative',@sigmoidDx) % Hidden Layer
    struct('size',5,'function',@sigmoid,'derivative',@sigmoidDx)
    struct('size',5,'function',@identity,'derivative',@identityDx)
]);


disp('Pesi W1');
disp(net.W{1});
disp('Bias liv. 1');
disp(net.b{1});
disp('Pesi W2');
disp(net.W{2});
disp('Bias output');
disp(net.b{2});

[outputs, A] = forwardProp(net, [1,2,3,4,5], false);

disp('A strato interno');
disp(A{1});
disp('A output');
disp(A{2});
disp('Z1');
disp(outputs{1});
disp('Y');
disp(outputs{2});

disp('fine');

