function g = dmsilu(z)    % derivative of the msilu function
    sigma = 1 ./ (1 + exp(-z));
    d_sigma = sigma .* (1 - sigma); % derivative of the sigmoid function
    g = d_sigma .* (1 + z .* (1 - sigma)) - (z .* exp(-z.^2) / 2);
end