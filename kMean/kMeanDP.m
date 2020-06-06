% Edward Hong
% K Means Clustering

%% Intialization
clc, clear, close all
load("dataset\NBA_dat.mat")
NBA_MPG = NBA_dat(:,5);
NBA_PPG = NBA_dat(:,7);

%% DP Means method:
%%% initialization
DATA = [NBA_MPG, NBA_PPG];

DPMean(DATA, 44)
DPMean(DATA, 100)
DPMean(DATA, 450)

%% Implementaiton of DP Means method
function DPMean(DATA, LAMBDA)
    %%% initialization\
    [Ndata,~] = size(DATA);
    K = 1; % # cluster count
    Z = ones(1,Ndata); % label
    MU = mean(DATA,1); % means initialized at global center
    converged = false;
    threshold_lambda = false;
    t = 0;
    mean_change = false;

    while (converged == 0)
        t = t + 1;

        %%% Compute distance from current point to all currently existing clusters
        dist = sum(MU .* MU , 2) + sum(DATA' .* DATA', 1) - (2.* MU * DATA');
        min_dist = min(dist,[],1);

        %%% Compare min distance of the cluster distance list compares to LAMBDA
        if sum(min_dist > LAMBDA) > 0
            threshold_lambda = true;
            K = K + 1;
            MU = [MU; DATA(min_dist == max(min_dist),:)]; % New MU is the newest
        end
        
        %%% Recompute means per cluster
        dist = sum(MU .* MU , 2) + sum(DATA' .* DATA', 1) - (2.* MU * DATA');
        [~,Z] = min(dist,[],1);

        for i = 1:K
            new_mean = mean(DATA(Z == i,:),1);
            if (new_mean ~= MU(i,:))
                mean_change = true;
            end
            MU(i,:) = new_mean;
        end

        %%% Test for convergence
        if ((~threshold_lambda && ~mean_change) || t > 500)
            converged = true;
        else
            threshold_lambda = false;
            mean_change = false;
        end

        %%% Plot after convergence
        if (converged)
            figure; hold;
            cmap = colormap(parula(K)); % chooses random color to plot
            
            for j = 1:K
                plot(MU(j,1),MU(j,2),'Marker','square','MarkerEdgeColor','k','MarkerFaceColor',cmap(j,:),'MarkerSize',12)
                scatter(DATA(Z == j,1),DATA(Z == j,2),'MarkerEdgeColor',cmap(j,:))
            end
            title(['DP-Mean on NBA Dataset for LAMBDA = ',num2str(LAMBDA)])
            xlabel('MPG')
            ylabel('PPG')
        end    
    end
end

