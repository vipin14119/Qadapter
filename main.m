%% Init Driver Code %%
clc;
clear;
%% Begins %%
disp('------------ Preparing data -------------');
data = load('./datasets/levelsData2.txt');
X = data(:, [1 2]);
y = data(:, 3);
levels = [0:1];
plotData(X, y, levels);
X = featureNormalize(X);
X = [ ones(size(X,1),1) X ];
theta = zeros(size(X,2),1);
[J, grad] = costFunction(X, y, theta)
%% Using Function minimization unconstrained
options = optimset('GradObj', 'on', 'MaxIter', 400);
[th, cost] = fminunc(@(t)(costFunction(X,y,t)), theta, options)

%% Gradient Descent %%
alpha = 0.01;
[theta, J_hist] = gradientDescent(X, y, theta, alpha);
iters = [1:400];
figure;
plot(iters, J_hist);