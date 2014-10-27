
function W = weight_init(n_hidden_nodes, n_feature, multiplier )
    %% weight initilation function
    W = cell(2,1);
    W{1} = (rand(n_hidden_nodes +1 ,n_feature + 1) - 0.5)./n_feature;
    W{1} = W{1} * multiplier;
    W{2} = (rand(1,n_hidden_nodes + 1) - 0.5)./n_hidden_nodes;
    W{2} = W{2} * multiplier;
 
end