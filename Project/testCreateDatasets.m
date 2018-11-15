[trainImages, trainLabels] = loadMNIST('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');
[testImages, testLabels] = loadMNIST('./mnist/t10k-images-idx3-ubyte', './mnist/t10k-labels-idx1-ubyte');

TRAINING_SET_SIZE = 100;
VALIDATION_SET_SIZE = 100; 
TEST_SET_SIZE = 100;
[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = createSets(trainImages', trainLabels, testImages', testLabels, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE);
testSetData = testSetData';

display_network(testSetData(:,1:100)); % Show the first 100 images
disp(testSetLabels(1:10));