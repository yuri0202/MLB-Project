[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

TRAINING_SET_SIZE = 100;
VALIDATION_SET_SIZE = 100; 
TEST_SET_SIZE = 100;
[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = createSets(trainImages', trainLabels, testImages', testLabels, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE);

shuffle   = randperm(100);
trainingSetData   = trainingSetData(shuffle,:);
trainingSetLabels = trainingSetLabels(shuffle,:);


%net = createNeuralNetwork(size(trainingSetData,2), 10, @identity, @identityDx, [
    %struct('size',320,'function',@sigmoid,'derivative',@sigmoidDx) % Hidden Layer
%]);

%[outputs, A] = forwardProp(net, trainingSetData, false);

%[dW, db] = backProp(net,trainingSetData, outputs,A,trainingSetLabels, @crossEntropyDx);