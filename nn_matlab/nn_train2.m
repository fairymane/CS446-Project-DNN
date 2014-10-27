function [c, W] = nn_train2(W, n_hidden_nodes, n_feature, h_act, ...
    train_data, train_label, lamda, n_iter, multiplier)
    % n_hidden_nodes:   number of hidden nodes
    % n_feature:        number of features for input examples
    % h_act:            store activation funtions in forward/backward process
    % train_data
    % train_label
    % lamda:            learning rate
    % n_iter:           number of iteration of forward/backward
    % multiplier:       parameter to specify the weight initilization range
    
    
    [y, h_act, X1] = forward_prop(W, h_act, train_data, 1);
    %c = zeros(1, n_iter);
  
    for i = 1: n_iter 
        %c(i)= cost(y, train_label);
        
        W = back_prop(y, train_label, W, h_act, lamda, X1, n_hidden_nodes, n_feature);
        [y, h_act, X1] = forward_prop(W, h_act, train_data, 0);
        
    end
    
    c = cost(y, train_label)
        
end

function c = cost(y, t)
c = dot(y-t, (y-t)');
%c
end