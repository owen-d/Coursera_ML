function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    coeff = (alpha / m);
    h = (X * theta - y);
    % X = n x 2 matrix
    % theta = 2 dimensional vector
    % h = n dimensional vector.
    % since X = (n x 2), X' = (2 x n), with the per-training example x-values being on each column
    % This allows us to multiple X' by h: a (2xn) matrix and a n dimensional vector. 
    % Multiplying them yields the x-value multiplied by the corresponding h(x)-y value. Then, everything is summed and divided by m. 
    theta = theta - coeff * X' * h;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
