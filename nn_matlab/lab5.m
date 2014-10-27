function lab5
format long 
[y, x1, x2] = textread('lab5data.txt', '%d %f %f');

X= horzcat(x1,x2);



[n_sample, n_feature] = size(X);
n_hidden_nodes = 32;


X = horzcat(ones(n_sample, 1), X);


% part a
figure(1)
X_p = X(find(y == 1), :);
X_n = X(find(y == -1), :);
scatter(X_p(:,1), X_p(:,2), 'r');
hold on
scatter(X_n(:,1), X_n(:,2), 'g');
hold off





% part b
figure(2);
W = weight_init(n_hidden_nodes, n_feature, 1);
[accr, y_rand] = nn_test(W, X, y);
X_p = X((y_rand == 1), :);
X_n = X((y_rand == -1), :);
err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
axis([0, 1, -1, 1]);
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('Random initialized weight, error rate = %f', err_rate);
title(title_str);
hold off


figure(3);
h_act = cell(2, 1);
n_iter = 1000;
lamda = 0.0001;
[c, W] = nn_train(n_hidden_nodes, n_feature, h_act, ...
        X, y,lamda, n_iter, 1);
[accr, y_t] = nn_test(W, X, y);
X_p = X((y_t == 1), :);
X_n = X((y_t == -1), :);

err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('NN training weight, error rate = %f', err_rate);
title(title_str);
hold off


%part c
h_act = cell(2, 1);
n_iter = 5000;
lamda = 0.0001;
K = 1./(1:n_iter);

[c,c_opt,W_opt] = nn_train_annealing(n_hidden_nodes, n_feature, h_act, ...
        X, y,lamda, n_iter, 1, K);
    
 plot([1:n_iter+1], c);
 %c_opt
  %c_opt - 2.21488321
  %c_opt = 1000 .* [2.214732422680314   2.226422496209246   2.226440870311030   2.226455058446218   2.226460144976341]
opt_size = length(c_opt); 

WW_opt = cell(opt_size, 1);
cc_opt = zeros(1,opt_size);

for i = 1: opt_size-1
  [c, W]= nn_train2(W_opt{i}, n_hidden_nodes, n_feature, h_act, ...
        X, y, 0.0001, 300, 1);
   
  i  
  WW_opt{i} = W;
  cc_opt(1, i) = c;
  
end

c_opt
cc_opt

figure(4);
[accr, y_t] = nn_test(WW_opt{1}, X, y);
X_p = X((y_t == 1), :);
X_n = X((y_t == -1), :);

err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('NN training weight, first local optimal,  error rate = %f', err_rate);
title(title_str);
hold off


figure(5);
[accr, y_t] = nn_test(WW_opt{2}, X, y);
X_p = X((y_t == 1), :);
X_n = X((y_t == -1), :);

err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('NN training weight, second local optimal,  error rate = %f', err_rate);
title(title_str);
hold off

figure(6);
[accr, y_t] = nn_test(WW_opt{3}, X, y);
X_p = X((y_t == 1), :);
X_n = X((y_t == -1), :);

err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('NN training weight, third local optimal,  error rate = %f', err_rate);
title(title_str);
hold off

figure(7);
[accr, y_t] = nn_test(WW_opt{4}, X, y);
X_p = X((y_t == 1), :);
X_n = X((y_t == -1), :);

err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('NN training weight, forth local optimal,  error rate = %f', err_rate);
title(title_str);
hold off


figure(8);
[accr, y_t] = nn_test(WW_opt{5}, X, y);
X_p = X((y_t == 1), :);
X_n = X((y_t == -1), :);

err_rate = 1 - accr;
scatter(X_p(:,2), X_p(:,3), 'r');
hold on
scatter(X_n(:,2), X_n(:,3), 'g');
title_str = sprintf('NN training weight, fifth local optimal,  error rate = %f', err_rate);
title(title_str);
hold off

    
