%% Init Driver Code %%
clc;
clear;
%% Begins %%

disp('------------ Preparing data -------------');
data = load('./datasets/levelsData1.txt');
X = data(:, [1 2]);
y = data(:, 3);
levels = [1:8];
plotData(X, y, levels);
[X_norm, mu, sigma]= featureNormalize(X);


%% Manipulation for y according to classes present in data %%

X = [ ones(size(X,1),1) X_norm ];
theta = zeros(size(X,2),1);
thetas = zeros(length(levels), length(theta));
costs = zeros(length(levels),1);
for l=levels
    fprintf('------- Using Level %d --------\n', l)
    y_new = y == l; % got logical y
    [J, grad] = costFunction(X, y_new, theta);
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = fminunc(@(t)(costFunction(X,y_new,t)), theta, options);
    thetas(l,:) = theta';
    costs(l,1) = cost;
    fprintf('Cost at above theta is %f\n', cost);
%     pos = find(y_new == 1);
%     neg = find(y_new == 0);
%     figure;hold on;
%     plot(X(pos,1), X(pos,2), '+', 'MarkerSize', 7, 'LineWidth', 2 );
%     plot(X(neg,1), X(neg,2), 'o', 'MarkerSize', 7, 'LineWidth', 2 );
%     hold off;
end
%% Training Data Accuracy %%

predicted_level = predict(X_norm,thetas, levels);
fprintf('Train Accuracy: %f\n', mean(double(predicted_level == y)) * 100);

%% Test Set %%

X = load('datasets/testSet.txt');
size(X)
mu = mu(1:size(X,1),:);
sigma = sigma(1:size(X,1),:);
X = (X - mu)./sigma;
predicted_level = predict(X,thetas, levels)
