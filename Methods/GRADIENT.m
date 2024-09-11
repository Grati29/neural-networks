function [losses, loss, grad_norms, test_loss, times, A_output_test] = GRADIENT (A_train, A_test, b_train, b_test, W_hidden, W_output)

    max_iter = 1000;  
    alpha = 1;      % constant step
    epsilon = 1e-8;  
    iter = 0;
    prev_loss = Inf;

    % colecting data trough iterations
    losses = zeros(max_iter, 1);
    times = zeros(max_iter, 1);
    grad_norms = zeros(max_iter, 1);

    while iter < max_iter
        tic
        iter = iter + 1;
 
        % hidden layer
        Z_hidden = A_train * W_hidden;
        A_hidden = msilu(Z_hidden);  % data activation using MSILU
        A_hidden = [ones(size(A_hidden, 1), 1) A_hidden]; 
    
        % stratul de iesire
        Z_output = A_hidden * W_output;
        A_output = sigmoid(Z_output);  % data activation using sigmoid for binary classification
    
        loss = binaryCrossEntropyLoss(b_train, A_output);
        % if it has converged
        if abs(prev_loss - loss) < epsilon
            break;
        end
        prev_loss = loss;
    
        % gradient calculation
        delta_output = A_output - b_train;
        grad_W_output = A_hidden' * delta_output / size(A_train, 1);
    
        delta_hidden = (delta_output * W_output(2:end, :)' .* dmsilu(Z_hidden));
        grad_W_hidden = A_train' * delta_hidden / size(A_train, 1);
    
        % update
        W_output = W_output - alpha * grad_W_output;
        W_hidden = W_hidden - alpha * grad_W_hidden;

        % colecting data
        grad_norms(iter) = norm([grad_W_output; grad_W_hidden(:)]);
        losses(iter) = loss;
        times(iter) = toc;
    end
    % testing
    % hidden layer
    Z_hidden_test = A_test * W_hidden;
    A_hidden_test = msilu(Z_hidden_test);
    A_hidden_test = [ones(size(A_hidden_test, 1), 1) A_hidden_test];

    % output layer
    Z_output_test = A_hidden_test * W_output;
    A_output_test = sigmoid(Z_output_test);

    % loss on testing data
    test_loss = binaryCrossEntropyLoss(b_test, A_output_test);
end