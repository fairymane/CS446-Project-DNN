function runlab
%out = load('diabetes_normalized', '-mat');
[y, x1, x2] = textread('lab5data.txt', '%d %f %f');

X = vertcat(x1', x2');
y = y';
[n_feature, n_sample] = size(X);

X = [ones(1,n_sample); X];

% two layer network
nodes_option = [1, 8, 32, 128];
h_act = cell(2, 1);
n_iter = 1000;
lamda = 0.0001;
x = [1:n_iter];
figure(1);

    [c, W] = nn_train(32 , n_feature, h_act, ...
        X, y, lamda, n_iter, 1);
    
    plot(x, c);
    accr = nn_test(W, X, y)

%{
figure(2);

test_err = zeros(4, 10);
train_acc = zeros(4, 10);

for f = 1:10
    for i = 1: 4
        [c, W] = nn_train(nodes_option(i), n_feature, h_act, ...
        train_data, train_label,lamda, n_iter, 1);
        train_err(i,f) = 1 - nn_test(W, train_data, train_label);
        test_err(i,f) = 1 - nn_test(W, test_data, test_label);
        
    end
end
e_test = mean(test_err, 2);
e_train = mean(train_err, 2);
plot(nodes_option, e_test);
hold on
grid on
plot(nodes_option, e_train, 'r');
%acc_rate = nn_test(W, test_data, test_label);

xlabel('num of hidden nodes');
ylabel('trainig/test error');

title('10-fold average trainig/test error');
legend('e_train','e_test')



figure(3);
x = [1:n_iter];
for i = 1: 4
    [c, W] = nn_train(nodes_option(i), n_feature, h_act, ...
        train_data, train_label,lamda, n_iter, 100);
    subplot(2,2,i);
    plot(x, c);
    xlabel('num of iteration');
    ylabel('cost');
    str = sprintf('num of hidden nodes = %d with big initial value ',nodes_option(i));
    title(str);

end



figure(4);

test_err = zeros(4, 10);
train_acc = zeros(4, 10);

for f = 1:10
    for i = 1: 4
        [c, W] = nn_train(nodes_option(i), n_feature, h_act, ...
        train_data, train_label,lamda, n_iter, 100);
        train_err(i,f) = 1 - nn_test(W, train_data, train_label);
        test_err(i,f) = 1 - nn_test(W, test_data, test_label);
    end
end
e_test = mean(test_err, 2);
e_train = mean(train_err, 2);
plot(nodes_option, e_test);
hold on
grid on
plot(nodes_option, e_train, 'r');
%acc_rate = nn_test(W, test_data, test_label);

xlabel('num of hidden nodes');
ylabel('trainig/test error');

title('10-fold average trainig/test error');
legend('e_train','e_test')



figure(5);
x = [1:n_iter];
for i = 1: 4
    [c, W] = nn_train(nodes_option(i), n_feature, h_act, ...
        train_data, train_label,lamda, n_iter, 0);
    subplot(2,2,i);
    plot(x, c);
    xlabel('num of iteration');
    ylabel('cost');
    str = sprintf('num of hidden nodes = %d with big initial value ',nodes_option(i));
    title(str);

end


figure(6);
e_test = zeros(1,4);
e_train = zeros(1,4);

for i = 1: 4
    [c, W] = nn_train(nodes_option(i), n_feature, h_act, ...
    train_data, train_label,lamda, n_iter, 0);
    e_train(1,i) = 1 - nn_test(W, train_data, train_label);
    e_test(1,i) = 1 - nn_test(W, test_data, test_label);
end

plot(nodes_option, e_test);
hold on
grid on
plot(nodes_option, e_train, 'r');
%acc_rate = nn_test(W, test_data, test_label);

xlabel('num of hidden nodes');
ylabel('trainig/test error');

title('10-fold average trainig/test error');
legend('e_train','e_test')

%}

end

%% weight initilation function
function W = weight_init(n_hidden_nodes, n_feature, multiplier )
    W = cell(2,1);
    W{1} = (rand(n_hidden_nodes +1 ,n_feature + 1) - 0.5)./n_feature;
    W{1} = W{1} * multiplier;
    W{2} = (rand(1,n_hidden_nodes + 1) - 0.5)./n_hidden_nodes;
    W{2} = W{2} * multiplier;
 
end

%% 
function [c, W] = nn_train(n_hidden_nodes, n_feature, h_act, ...
    train_data, train_label, lamda, n_iter, multiplier)

    W = weight_init(n_hidden_nodes, n_feature, multiplier);
    
    [y, h_act, X1] = forward_prop(W, h_act, train_data, 1);
    
    c = zeros(1, n_iter);
  
    for i = 1: n_iter 
        c(i)= cost(y, train_label);
        c(i)
        W = back_prop(y, train_label, W, h_act, lamda, X1, n_hidden_nodes, n_feature);
        norm(W{1})
        norm(W{2})
        
        [y, h_act, X1] = forward_prop(W, h_act, train_data, 0);
    end
    
        
end

%% testing function
function acc_rate = nn_test (W, test_data, test_label)
test_h_act = cell(2, 1);

[y,test_h_act, X1] = forward_prop(W, test_h_act, test_data, 0);

y_t = sign(y);
n_error = nnz(y_t - test_label);

acc_rate = 1- n_error / length(test_label);

end

%% forward prop
function [y,h_act, X1] = forward_prop(W, h_act, X, init)
    %Calcuate forward prop
    
    %layer 1
    X1 = cell(2, 1);
    
    %size(W{1})
    %size (X)
    
    h_act{1} = W{1} * X; % 5 * 500
    
    if init == 1
        length(h_act{1});
        h_act{1}(1, :) = ones(1,length(h_act{1})); % 5 * 500
    end
    X1{2} = tanh(h_act{1}); % 5 * 500
    
    %layer 2 .i.e. output y
    h_act{2} = W{2} * h_act{1}; %1 * 500
    y = tanh (h_act{2});
    X1{1} = X; % 9 * 500
    
end

function c = cost(y, t)
c = dot(y-t, (y-t)');
%c
end

%% first order tanh derivitive
function value = hyper_tangent_1d(a)

value = ones(size(a)) - tanh(a).*tanh(a);
end


function W = back_prop(y, t, W, h_act, lamda, X1, n_hidden_nodes, n_feature)

delta = cell(2,1);
SD = cell(2, 1);
    for l = 2: -1 : 0
        %disp(1)
        if (l == 2)
            %size( hyper_tangent_1d(h_act{l}) )
            %size(2 * (y - t))
            delta{l} =  hyper_tangent_1d(h_act{l}) .* (2 * (y - t));  % g'(a) .* d (y-t)^2 on layer L -- delta{3} 1*500
    
        elseif l == 1
            

            delta{l} = W{l+1}' * delta{l+1} .* hyper_tangent_1d(h_act{l}) ; % W{2}' 5*1, delta{3} 1 * 500, h_act{1}, 1 *500 -- delta{2} 5*500
        
            %size(delta{l})
            %size(X1{l+1})
            SD{l+1} =  sum(delta{l} .* X1{l+1}, 2); % 1 * 5!
            
            %size(W{l+1})
            %size(SD{l+1})
            W{l+1} = W{l+1} - lamda * SD{l+1}';
        elseif l == 0
            %disp(l)
        
            sd = zeros(n_hidden_nodes+1, n_feature+1 );
            for i = 1:size(delta{l+1},2)
                sd = sd + delta{l+1}(:, i) * X1{l+1}(:,i)';
            end
        
            W{l+1} =  W{l+1} - lamda * sd; 
        end
    end
    
end



