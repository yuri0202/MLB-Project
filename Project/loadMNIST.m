function [ mnistImages, mnistLabels ] = loadMNIST( imagesFilename, labelsFilename )
% LoadMNIST legge e importa le immagini e le labels del dataset MNIST
% INPUT:
%   - 'imagesFilename': Percorso e nome del file delle immagini
%   - 'labelsFilename': Percorso e nome del file delle labels
% OUTPUT:
%   - 'minstImages': Matrice di dimensione 28x28x[numero di immagini], che
%   contiene le immagini grezze del dataset MNIST
%   - 'mnistLabels': Array di dimensione [numero di immagini], che
%   contiene le labels delle immagini del dataset MNIST
    mnistImages = loadMNISTImages(imagesFilename);
    mnistLabels = loadMNISTLabels(labelsFilename);
end


function images = loadMNISTImages(filename)
% loadMNISTImages ritorna una matrice di dimensione 28x28x[numero di immagini]
% che contiene le immagini grezze del dataset MNIST

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


% Ridimensiona in #pixels x #esempi
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Converti in double e riscala in [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels ritorna un array di dimensione [numero di immagini], che
%contiene le labels delle immagini del dataset MNIST
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end

