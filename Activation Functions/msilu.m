function y = msilu(z)       % activation function
    sigma = 1 ./ (1 + exp(-z));
    y = z .* sigma + (exp(-z.^2) - 1) / 4;
end