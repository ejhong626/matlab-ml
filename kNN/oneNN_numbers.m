% Edward Hong
% Nearest Neighbor Classifier with Batch Learning

%% Intialization
clc, clear
load("dataset\data_mnist_train.mat");
load("dataset\data_mnist_test.mat");

[Ntrain, dims] = size(X_train);
[Ntest, ~] = size(X_test);

batch_size = 500;  % the dataset is 60000x784, needs to be split or the RAM can't handle it
num_batches = Ntest / batch_size;


%% Algorithm
ypred = zeros(size(Y_test));

for bn = 1:num_batches
    batch_start = 1 + (bn - 1) * batch_size;
    batch_stop = batch_start + batch_size - 1;

    fprintf("1-NN classification for batch %d\n", bn);

    y_transpose = X_train(1:60000,:)';
    x_term = sum(X_test(batch_start:batch_stop,:) .* X_test(batch_start:batch_stop,:) , 2);
    y_term = sum(y_transpose .* y_transpose, 1);
    cross_term = 2.*X_test(batch_start:batch_stop,:) * y_transpose;  
    dist = x_term + y_term - cross_term;
    
    [eu_min,indx] = min(dist,[],2); % find indices, indx, for minimums to help find the corresponding label
    ypred(batch_start:batch_stop) = Y_train(indx);
end

%% Compute Results
conf_mat = confusionmat(Y_test, ypred);
ccr = sum(diag(conf_mat))/Ntest;
fprintf("1-NN classification CCR %3.2f%%\n", ccr * 100);

