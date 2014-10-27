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
            
            %size(W{l+1}')
            %size(delta{l+1}')
            %size(hyper_tangent_1d(h_act{l}))
            % W{l+1}' * 
            delta{l} = W{l+1}' * delta{l+1}' .* hyper_tangent_1d(h_act{l})' ; % W{2}' 5*1, delta{3} 1 * 500, h_act{1}, 1 *500 -- delta{2} 5*500
            %size(delta{l})
            
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

function value = hyper_tangent_1d(a)
%% first order derivitive of tanh 
value = ones(size(a)) - tanh(a).*tanh(a);
end