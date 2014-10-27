
function [y,h_act, X1] = forward_prop(W, h_act, X, init)
    %Calcuate forward prop
    
    %layer 1
    X1 = cell(2, 1);  
    %size(W{1})
    %size(X')
    h_act{1} = W{1} * X'; 
    
    if init == 1
        length(h_act{1});
        h_act{1}(1, :) = ones(1,length(h_act{1})); 
    end
    X1{2} = tanh(h_act{1}); 
    
    %layer 2 .i.e. output y
    %size(W{2})
    h_act{2} = W{2} * h_act{1}; 
    %size(h_act{2})
    y = tanh (h_act{2});
    h_act{1} = h_act{1}';
    h_act{2} = h_act{2}';
    y = y';
    X1{1} = X'; 
    
end