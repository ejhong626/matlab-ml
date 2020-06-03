% Edward Hong
% Nearest Neighbor Classifier

%% Intialization
clc, clear
load("data_knnSimulation.mat")

%% Plotting Dataset
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain)
legend('Class 1','Class 2','Class 3')
xlabel('X Axis')
ylabel('Y Axis')
title('Data Scatterplot')

%% Plotting Heatmap Probabilities on a 2D map
heatmap_kNN(Xtrain, ytrain, 10, 2)
heatmap_kNN(Xtrain, ytrain, 10, 3)

%% Class Label Predictions
labelmap_kNN(Xtrain, ytrain, 1)
labelmap_kNN(Xtrain, ytrain, 5)

%% LOOCV CCR Computations

for k = 1:2:11 % only odd kNN, even would run into tie-breaker issue and affects CCR
    ypred = zeros(size(ytrain));
    for i = 1:size(ypred)
        % setup for leave one out 
        LOO_Xtest = Xtrain(i,:); 
        LOO_Xtrain = Xtrain;
        LOO_Xtrain(i,:) = [];
        LOO_ytrain = ytrain;
        LOO_ytrain(i,:) = [];
        
        % compute dist to leave one out
        dist = [LOO_Xtrain(:,1)-LOO_Xtest(1,1), LOO_Xtrain(:,2)-LOO_Xtest(1,2)].^2;
        eu_dist = sortrows([sum(dist,2),LOO_ytrain]);
        ypred(i) = mode(eu_dist(1:k,2));
    end

    % compute confusion matrix & CCR
    conf_mat = confusionmat(ytrain, ypred);
    CCR = sum(diag(conf_mat))/size(Xtrain,1);
    
    % Logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% Plot result
figure
plot(1:2:11,CCR_values);
title('CCR Values vs k Values');
xlabel('k Values')
ylabel('CCR Values')

%% Functions
function heatmap_kNN(Xtrain, ytrain, kN, label)
    % Initialize probility mesh grid (heatmap)
    [Xgrid, Ygrid] = meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
    Xtest = [Xgrid(:),Ygrid(:)];
    [Ntest,~]=size(Xtest);
    probabilities = zeros(Ntest,1); % 9216 => size of Xtest
    
    % Compute NN
    for i = 1:Ntest
        dist = [Xtrain(:,1)-Xtest(i,1), Xtrain(:,2)-Xtest(i,2)].^2;                 % compute euclidean distance of 1 grid point to all points
        eu_dist = sortrows([sqrt(sum(dist,2)),ytrain]);                                    % sort by label and find 10 nearest
        probabilities(i) = sum(eu_dist(1:kN,2)==label)/kN;
    end
    
    % Plot
    figure;
    classProbOnGrid = reshape(probabilities,size(Xgrid));
    contourf(Xgrid,Ygrid,classProbOnGrid);
    colorbar;

    xlabel('X Axis')
    ylabel('Y Axis')
    title(sprintf('Probility of %dNN(y = %d|x) Heat Map', kN, label))
end

function labelmap_kNN(Xtrain, ytrain, kN)
    % Intialize grid
    [Xgrid, Ygrid] = meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
    Xtest = [Xgrid(:),Ygrid(:)];
    [Ntest,~]=size(Xtest);
    ypred = zeros(9216,1);
    
    % Compute predictions
    for i = 1:Ntest
        dist = [Xtrain(:,1)-Xtest(i,1), Xtrain(:,2)-Xtest(i,2)].^2;                 % compute euclidean distance of 1 grid point to all points
        eu_dist = sortrows([sum(dist,2),ytrain]);                                    % sort by label and find 10 nearest
        ypred(i) = mode(eu_dist(1:kN,2)); 
    end
    
    % Plot
    figure
    gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
    xlim([-3.5,6]);
    ylim([-3,6.5]);
    xlabel('X Axis')
    ylabel('Y Axis')
    title(sprintf('h%dNN(x) classifier decision regions', kN))
end