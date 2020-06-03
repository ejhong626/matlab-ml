% Edward Hong
% K Means Clustering

%% Intialization
clc, clear
% generating 3 gaussian clusters
data_1 = [normrnd(2,sqrt(0.02),1,50);normrnd(2,sqrt(0.02),1,50)];
data_2 = [normrnd(-2,sqrt(0.05),1,50);normrnd(2,sqrt(0.05),1,50)];
data_3 = [normrnd(0,sqrt(0.07),1,50);normrnd(-3.25,sqrt(0.07),1,50)];
DATA = [data_1,data_2,data_3];

%% Plotting Dataset
figure; hold;
scatter(data_1(1,:),data_1(2,:),'r')
scatter(data_2(1,:),data_2(2,:),'g')
scatter(data_3(1,:),data_3(2,:),'b')
title('3 Gaussian Clusters')
xlabel('X')
ylabel('Y')

%% Clustering N clusters
kMean(DATA, 3, 0.025) % # of cluster mean = clusters
kMean(DATA, 10, 0.025) % # of cluster mean > clusters

%% Algorithm Implementaiton
function kMean(DATA, N, threshold)
    %%% initializations
    [~, nData] = size(DATA);
    fin_labels = zeros(nData,1);
    fin_MU = zeros(N,2); % Will save all MU initial and its final
    M = 10; % # of trials
    
    for i = 1:N % repeat N clusters
        WCSS = Inf; % initialize as INF to find minimum
        for j = 1:M % repeats for M trials
            % initializations
            converged = false;
            MU_init = zeros(i,2);

            for k = 1:i
                MU_init(k,:) = DATA(:,randperm(nData,1))';
            end
            MU_previous = MU_init;
            MU_current = MU_init;
            
            while (~converged)
                %%% each data observation assigned to a cluster with nearest mean:
                dist = sum(MU_current .* MU_current , 2) + sum(DATA .* DATA, 1) - (2.*MU_current * DATA);
                [min_dist,indx ] = min(dist,[],1);

                %%% update the cluster means
                MU_previous = MU_current;
                for k = 1:i
                    MU_current(k,:) = mean(DATA(:,indx' == k),2)';
                end

                %%% check for convergence threshold
                per_change = (MU_current - MU_previous)./(MU_previous);
                if (abs(max(per_change,[],'all')) < threshold)
                    converged = true;
                end

                %%% plot clustering results if converged:
                if (converged)
                    %%% get WCSS metric
                    if (sum(min_dist) < WCSS)
                        WCSS = sum(min_dist);
                        fin_labels = indx';
                        fin_MU = MU_current;
                    end
                end
            end
        end
    end

    %%% Plot
    cmap = colormap(parula(N)); % random colors
    c_shuffle = randperm(N);
    
    figure; hold;
    for i = 1:N
        plot(fin_MU(i,1),fin_MU(i,2),'Marker','square','MarkerEdgeColor','k','MarkerFaceColor',cmap(c_shuffle(i),:),'MarkerSize',12)
        scatter(DATA(1,fin_labels == i),DATA(2,fin_labels == i),'MarkerEdgeColor',cmap(c_shuffle(i),:))
    end
    title(sprintf('kMeans clustering for %d clusters', N))
    xlabel('X')
    ylabel('Y')
end
