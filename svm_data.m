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
[classifier,validacc]=trainClassifier(trainingdata3);
pred=classifier.predictFcn(X_norm);
f1=MyClassifyPerf(Y,pred);
disp(f1.F1);
