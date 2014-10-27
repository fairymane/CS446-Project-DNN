function [c, c_opt,W_opt] = nn_train_annealing(n_hidden_nodes, n_feature, h_act, ...
    train_data, train_label, lamda, n_iter, multiplier, K)
    % n_hidden_nodes:   number of hidden nodes
    % n_feature:        number of features for input examples
    % h_act:            store activation funtions in forward/backward process
    % train_data
    % train_label
    % lamda:            learning rate
    % n_iter:           number of iteration of forward/backward
    % multiplier:       parameter to specify the weight initilization range
    
    W = weight_init(n_hidden_nodes, n_feature, multiplier);

    
    
    [y, h_act, X1] = forward_prop(W, h_act, train_data, 1);
    
    c = zeros(1, n_iter+1);
    count = 6;
    c_opt = zeros(1, count);
    W_opt = cell(count,1);
    c(1)= cost(y, train_label);
  
    flag = 0;
    count_i = 1;
    for i = 1: n_iter 
        
        W = back_prop_annealing(y, train_label, W, h_act, lamda, X1, n_hidden_nodes, n_feature, K(i) );
        %w1_n = norm(W{1})
        %w2_n = norm(W{2})
        
        [y, h_act, X1] = forward_prop(W, h_act, train_data, 0);
        c(i+1)= cost(y, train_label);
        %c(i+1)
        if c(i+1) < c(i) & flag == 0 %& i > 2000
            flag = 1;
 
        end
        if c(i+1) > c(i) & flag == 1 %& i > 2000
            c_opt(count_i) = c(i);
            W_opt{count_i} = W;
            flag = 0;
            count_i = count_i+1;
        end
        if count_i == count
            break;
        end
    end    
        
end

function c = cost(y, t)
c = dot(y-t, (y-t)');
%c
end