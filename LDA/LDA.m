% Edward Hong
% Linear Discriminant Analysis

%% Intialization
clc, clear

%% Generate Data
n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];
[X, Y] = gaussian2D(n1, n2, mu1, mu2, 0.25, 1, pi/6);
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure, hold, grid, axis equal;
scatter(X1(1,:),X1(2,:),'o','fill','b');
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(0),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;


%% Compute SNR along phi, plot them against phi
phi_array = 0:pi/48:pi;
signal_power_arr = zeros(1,length(phi_array));
noise_power_arr = zeros(1,length(phi_array));
snr_arr = zeros(1,length(phi_array));

%%% Loop through all of phi
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = snrMetric(X, Y, phi_array(i), false);
    signal_power_arr(i) = signal_power;
    noise_power_arr(i) = noise_power;
    snr_arr(i) = snr;
end

%%% Finding Max/Min
[~,S_max] = max(signal_power_arr);
[~,N_min] = min(noise_power_arr);
[~,SNR_max] = max(snr_arr);

%%% Subplots Setup
subP_titles = {'Phi vs Signal Strength', 'Phi vs Noise Strength', 'Phi vs SNR'};
subP_labels = {'Signal', 'Noise', 'SNR'};
subP_plot = {signal_power_arr, noise_power_arr, snr_arr};
subP_ind = {S_max, N_min, SNR_max};
figure

%%% Plot Subplot
for i = 1:3
    subplot(3,1,i);hold on;grid;
    plot(phi_array,subP_plot{i});
    scatter(phi_array(subP_ind{i}),subP_plot{i}(subP_ind{i}),'o','fill','b');
    xlabel('Phi');ylabel(subP_labels{i});
    title(subP_titles{i});
end

%%% Dataset is rotated by pi/6, should observe the overlapping density is
%%% least along pi/6
phi_arr = [0, pi/6, pi/3];
for i = 1:length(phi_arr)
    snrMetric(X, Y, phi_arr(i), true);
end

%% Solve LDA and Plot
w_LDA = LDA2D(X,Y);

mu1 = mean(X(:,Y==1),2);
mu2 = mean(X(:,Y==2),2);
mu_vect = mu2-mu1;
figure; hold;
gscatter(X(1,:),X(2,:),Y);
quiver(mu1(1,1),mu1(2,1),mu_vect(1,1),mu_vect(2,1),'LineWidth',4);
quiver(mu1(1,1),mu1(2,1),w_LDA(1,1),w_LDA(2,1),'LineWidth',4);
legend('Class 1','Class 2','(Mean2 - Mean1) Vector','wLDA')
xlabel('X')
ylabel('Y')
title('LDA Scatter')

%% CCR for LDA
n = length(X);
X_project = w_LDA' * X; 
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);

for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

[Value, Index] = max(ccr_array);
CCR_argmax = find(ccr_array == Value); % finds argmax
figure; hold; grid on;
scatter(b_array(CCR_argmax),ccr_array(CCR_argmax),'filled','r')
plot(b_array,ccr_array)
xlabel('b Value')
ylabel('CCR Value')
title('b vs CCR Value')
legend('Argmax of wx+b','b vs CCR')

%% Functions
function [X, Y] = gaussian2D(n1,n2,mu1,mu2,lambda1,lambda2,theta)
    % Generate 2 class dataset of labeled 2D points, points generate from
    % Gaussian distribution with same covariaance matrix but different mean

    Ntest = n1 + n2;
    X = zeros(2, Ntest);
    Y = zeros(1, Ntest);
    mu_mat = [mu1';mu2'];
    ortho_mat = [cos(theta),sin(theta);sin(theta),-cos(theta)];
    lambda_mat = eye(2) .* [lambda1;lambda2];
    cov_mat = ortho_mat * lambda_mat * ortho_mat';

    for i = 1:Ntest
        gauss_2d = mvnrnd(mu_mat,cov_mat);
        if i <= n1
            X(:,i) = gauss_2d(1,:)';
            Y(1,i) = 1;
        elseif i > n1
            X(:,i) = gauss_2d(2,:)';
            Y(1,i) = 2;
        end
    end
end

function [signal, noise, snr] = snrMetric(X, Y, phi, want_plot)
    % Returns the Signal, Noise, SNR of dataset along given direction phi
    % and plots the result and the class density estimation

    n = [length(find(Y==1)),length(find(Y==2)),length(X)];
    w = [cos(phi); sin(phi)];
    proj_X = ((X'*w) * w')'; % L2 norm ignored since w^2 = sin^2 + cos^2 = 1
    proj_mu = [mean(proj_X(:,Y==1),2)';mean(proj_X(:,Y==2),2)']; % class mean per row
    sX1 = ((proj_X(:,Y==1) - proj_mu(1,:)') * (proj_X(:,Y==1) - proj_mu(1,:)')')/n(1);
    sX2 = ((proj_X(:,Y==2) - proj_mu(2,:)') * (proj_X(:,Y==2) - proj_mu(2,:)')')/n(2);
    
    signal = (w'*diff(proj_mu)')^2;
    noise = (n(1)/n(3))*(w' * sX1 * w) * (n(2)/n(3))*(w' * sX2 * w);
    snr = signal/noise;
    X_projected_phi_class1 = (w'*proj_X(:,Y==1))'; % reduce 2D into 1D based on direction PHI
    X_projected_phi_class2 = (w'*proj_X(:,Y==2))';

    % Plot density estimates for both classes along chosen direction phi
    if (want_plot)
        figure, hold;
        [pdf1,z1] = ksdensity(X_projected_phi_class1);
        plot(z1,pdf1)
        [pdf2,z2] = ksdensity(X_projected_phi_class2);
        plot(z2,pdf2)
        grid on;
        hold off;
        legend('Class 1', 'Class 2')
        xlabel('projected value')
        ylabel('density estimate')
        title(['Estimated class density estimates of data projected along \phi = ',num2str(phi/(pi/6)),' \times \pi/6. Ground-truth \phi = \pi/6'])
    end
end

function w_LDA = LDA2D(X, Y)
    % Given dataset and label, the function returns the LDA solution

    n = length(Y);
    n1 = numel(find(Y==1));
    n2 = numel(find(Y==2));
    mu1 = mean(X(:,Y==1),2); 
    mu2 = mean(X(:,Y==2),2);
    s1 = ((X(:,Y==1) - mu1) * (X(:,Y==1) - mu1)')/n1;
    s2 = ((X(:,Y==2) - mu2) * (X(:,Y==2) - mu2)')/n2;
    main_S = ((n1/n)*s1 + (n2/n)*s2)^-1;
    w_LDA = main_S*(mu2 - mu1);
end

function ccr = compute_ccr(X, Y, w_LDA, b)
    % Computes CCR of LDA solution on given dataset
    label = zeros(1,length(X)); 
    w_LDA(2,1) = w_LDA(2,1) * (-1);
	h_x = w_LDA' * X + b;
    label(h_x <= 0) = 1;
    label(h_x > 0) = 2;
    
    conf_mat = confusionmat(Y, label);
    ccr = sum(diag(conf_mat))/length(X);
end