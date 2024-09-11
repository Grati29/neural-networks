function [losses, loss, grad_norms, test_loss, times, A_output_test] = BFGS (A_train, A_test, b_train, b_test, W_hidden, W_output)

    max_iter = 1000;    
    alpha = 0.1;     % constant step     
    epsilon = 1e-8;     
    H_inv = eye(size(W_output, 1));  

    % colecting data trough iterations
    losses = zeros(max_iter, 1);
    times = zeros(max_iter, 1);
    grad_norms = zeros(max_iter, 1);

    for iter = 1:max_iter
        tic; 
        % hidden layer
        Z_hidden = A_train * W_hidden;
        A_hidden = msilu(Z_hidden);  % data activation using MSILU
        A_hidden = [ones(size(A_hidden, 1), 1) A_hidden];  
    
        % output layer
        Z_output = A_hidden * W_output;
        A_output = sigmoid(Z_output);   % data activation using sigmoid for binary classification
    
        loss = binaryCrossEntropyLoss(b_train, A_output);
    
        % gradient calculation
        delta_output = A_output - b_train;
        grad_W_output = A_hidden' * delta_output / size(A_train, 1);
    
        % update
        W_output_old = W_output;
        grad_W_output_old = grad_W_output;
        p = -H_inv * grad_W_output; % direction
        W_output = W_output + alpha * p;
    
        Z_hidden = A_train * W_hidden;
        A_hidden = msilu(Z_hidden);  
        A_hidden = [ones(size(A_hidden, 1), 1) A_hidden];
    
        Z_output = A_hidden * W_output;
        A_output = sigmoid(Z_output);
    
        % new gradient
        delta_output = A_output - b_train;
        grad_W_output = A_hidden' * delta_output / size(A_train, 1);
    
        s = W_output - W_output_old;
        y = grad_W_output - grad_W_output_old;
        rho = 1 / (y' * s);
    
        % approximation of the inverse Hessian
        H_inv = (eye(size(H_inv)) - rho * (s * y')) * H_inv * (eye(size(H_inv)) - rho * (y * s')) + rho * (s * s');
    
        % colecting data
        grad_norms(iter) = norm(grad_W_output);
        losses(iter) = loss;
        times(iter) = toc; 
    
        % if it has converged
        if grad_norms(iter) < epsilon
            break;
        end
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