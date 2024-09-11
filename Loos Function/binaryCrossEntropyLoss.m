function loss = binaryCrossEntropyLoss(e, y)
    % loss function
    N = length(e);
    loss = -(1/N) * sum(e .* log(y) + (1 - e) .* log(1 - y));
end