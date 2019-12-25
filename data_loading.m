load('feature_data.txt');
X=feature_data(:,1:1140);
Y=feature_data(:,1141);
[X_norm, mu, sigma] = featureNormalize(X);

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

covar_mat = (X_norm' * X_norm) / m;
[U, S, V] = svd(covar_mat);
%a= find(all(isnan(covar_mat)));
%both X_norm and covar_mat has 60 rows with Nan;
U_reduced=U(:,1:50);
Z=X*U_reduced;
trainingdata=[Z Y];
trainingdata2=[X Y];
trainingdata3=[X_norm Y];


%For Neural Networks
training_nn=X;

%Target = [t_0;t_1;t_2];
target_n = full(ind2vec(Y'))';
data_n = [X target_n];

x = training_nn';
t = target_n';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)
tt = vec2ind(t);
yy = vec2ind(y);
c = confusionmat(tt,yy);
accuracy=trace(c)/sum(sum(c))

predictions=MyClassifyPerf(tt,yy);

