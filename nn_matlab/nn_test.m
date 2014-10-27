
function [acc_rate,y_t] = nn_test (W, test_data, test_label)
%% testing function
test_h_act = cell(2, 1);

[y,test_h_act, X1] = forward_prop(W, test_h_act, test_data, 0);

y_t = sign(y);
n_error = nnz(y_t - test_label);

acc_rate = 1- n_error / length(test_label);

end