function [ mnistImages, mnistLabels ] = loadMnist( imagesPath, labelsPath)
% Questa funzione carica il dataset Mnist, importando le immagini e le
% labels.
%   INPUT:
%       - 'imagesPath': Percorso e nome del file contenente le immagini
%       - 'labelsPath': Percorso e nome del file contenente le labels
%
%   OUTPUT:
%       - 'mnistImages': Matrice che contiene le immagini raw del dataset
%       Mnist
%       - 'mnistLabels': Array che contiene le label del dataset Mnist

    mnistImages = loadMNISTImages(imagesPath);
    mnistLabels = loadMNISTLabels(labelsPath);
end


function images = loadMNISTImages(filename)
% Questa funzione ritorna una matrice di dimensione 28x28x(#Immagini Mnist)
% contenente le immagini del dataset


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples TODO : Capire
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
% Questa funzione ritorna un array di dimensione pari al numero delle
% immagini nel dataset MNIST, che contiene le label di ogni immagine


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end

