% Script per testare la forward propagation

% Creo la rete neurale
net = createNeuralNetwork(2, 1, @identity, @identityDx, [
    struct('size',7,'function',@sigmoid,'derivative',@sigmoidDx) % Hidden Layer
]);


disp('Pesi W1');
disp(net.W{1});
disp('Bias liv. 1');
disp(net.b{1});
disp('Pesi W2');
disp(net.W{2});
disp('Bias output');
disp(net.b{2});

input = [1,2;1,5;3,4];
target = [1;2;1];
[outputs, A] = forwardProp(net, input, true);

disp('A strato interno');
disp(A{1});
disp('A output');
disp(A{2});
disp('Z1');
disp(outputs{1});
disp('Y');
disp(outputs{2});

[dw,db] = backProp(net,input,outputs,A,target,@crossEntropyDx);

