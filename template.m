% data loading and processing
opts = detectImportOptions('parkinsons.data', 'FileType', 'text', 'NumHeaderLines', 1);
opts.SelectedVariableNames = opts.VariableNames(2:end);
data = readtable('parkinsons.data', opts);
A = table2array(data(:, [1:16, 18:end]));        % converting the table columns into a matrix
b = table2array(data(:, 17));                    % the labels are in column 17
rng('default');

% data shuffling
shuffledIndices = randperm(size(A, 1));
shuffledA = A(shuffledIndices, :);
shuffledb = b(shuffledIndices);

% splitting the data into training and testing sets
numTrain = floor(0.8 * size(shuffledA, 1));      % 80% training data
A_train = shuffledA(1:numTrain, :);
b_train = shuffledb(1:numTrain);
A_test = shuffledA(numTrain+1:end, :);
b_test = shuffledb(numTrain+1:end);

% data normalization
A_train = normalize(A_train);
A_test = normalize(A_test);

n = size(A_train, 2);                            % number of features
m = 20;                                          % chosen number of neurons in the hidden layer

W_hidden = 0.01 * randn(n + 1, m);
W_output = 0.01 * randn(m + 1, 1);
A_train = [ones(size(A_train, 1), 1) A_train];
A_test = [ones(size(A_test, 1), 1) A_test];

% the method bfgs
[losses_1, loss_1, grad_norms_1, test_loss_1, times_1, A_output_test_1] = BFGS (A_train, A_test, b_train, b_test, W_hidden, W_output);
fprintf('Training with BFGS Method ended with the final loss function value of %.4f\n', loss_1);
time_1 = sum(times_1);
fprintf('Loss value on test data - BFGS: %.4f\n', test_loss_1);

[losses_2, loss_2, grad_norms_2, test_loss_2, times_2, A_output_test_2] = GRADIENT (A_train, A_test, b_train, b_test, W_hidden, W_output);
fprintf('Training with Gradient Method ended with the final loss function value of  %.4f\n', loss_2);
time_2 = sum(times_2);
fprintf('Loss value on test data - Gradient: %.4f\n', test_loss_2);

% plot of objective function vs iterations
figure;
subplot(2, 2, 1);
semilogy(losses_1, 'b-', 'LineWidth', 2);
hold on;
semilogy(losses_2, 'r--', 'LineWidth', 2);
title('objective function vs iterations');
xlabel('iterations');
ylabel('objective function');
legend('BFGS', 'Gradient');

% plot of objective function vs time
subplot(2, 2, 2);
plot(cumsum(times_1), losses_1, 'b-', 'LineWidth', 2);
hold on;
plot(cumsum(times_2), losses_2, 'r--', 'LineWidth', 2);
title('objective function vs time');
xlabel('time');
ylabel('objective function');
legend('BFGS', 'Gradient');

% plot of gradient norm vs iterations
subplot(2, 2, 3);
semilogy(grad_norms_1, 'b-', 'LineWidth', 2);
hold on;
semilogy(grad_norms_2, 'r--', 'LineWidth', 2);
title('gradient norm vs iterations');
xlabel('iterations');
ylabel('gradient norm');
legend('BFGS', 'Gradient');

% plot of gradient norm vs time
subplot(2, 2, 4);
plot(cumsum(times_1), grad_norms_1, 'b-', 'LineWidth', 2);
hold on;
plot(cumsum(times_2), grad_norms_2, 'r--', 'LineWidth', 2);
title('gradient norm vs time');
xlabel('time');
ylabel('gradient norm');
legend('BFGS', 'Gradient');

% model performance after training the network with the BFGS method

% prediction calculation
predictions_1 = A_output_test_1 > 0.5;                 % to make binary predictions
predictions_numeric_1 = double(predictions_1);           % prediction conversion

% confusion matrix calculation
conf_matrix_1 = confusionmat(b_test, predictions_numeric_1);
disp('confusion matrix BFGS:');
disp(conf_matrix_1);

% performance metrics calculation
TP_1 = conf_matrix_1(1, 1); 
TN_1 = conf_matrix_1(2, 2); 
FP_1 = conf_matrix_1(2, 1); 
FN_1 = conf_matrix_1(1, 2); 

sensitivity_1 = TP_1 / (TP_1 + FN_1);
accuracy_1 = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1);
fprintf('Sensitivity BFGS: %.4f\n', sensitivity_1);
fprintf('Accuracy BFGS: %.4f\n', accuracy_1);

%  model performance after training the network with the Gradient method
predictions_2= A_output_test_2 > 0.5; 
predictions_numeric_2 = double(predictions_2);

% confusion matrix calculation
conf_matrix_2 = confusionmat(b_test, predictions_numeric_2);

disp('confusion matrix Gradient method:');
disp(conf_matrix_2);

% performance metrics calculation
TP_2 = conf_matrix_2(1, 1); 
TN_2 = conf_matrix_2(2, 2); 
FP_2 = conf_matrix_2(2, 1); 
FN_2 = conf_matrix_2(1, 2);

sensitivity_2 = TP_2 / (TP_2 + FN_2);
accuracy_2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2);
fprintf('Sensitivity Gradient: %.4f\n', sensitivity_2);
fprintf('Accuracy Gradient: %.4f\n', accuracy_2);
