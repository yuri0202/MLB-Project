function plotErrors(trainingSetErrors, validationSetErrors)
% Questa funzione crea un semplice plot per mostrare l'andamento delle
% funzioni di errore calcolate sul Training e sul Validation Set
%
% INPUT:
%   - 'trainingSetErrors': Array di errori tale che l'i-esimo elemento 
%                          rappresenta l'errore sul training set alla
%                          i-esima epoca
%   - 'validationSetErrors': Array di errori tale che l'i-esimo elemento
%                            rappresenta l'errore sul validation set alla
%                            i-esima epoca

    % Calcolo dei grafici con annesa legenda
    plot(trainingSetErrors);
    hold
    plot(validationSetErrors);
    legend('Error on the Training Set', 'Error on the Validation Set');
end