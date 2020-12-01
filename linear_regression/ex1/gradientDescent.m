function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
k = 1:m;

for iter = 1:num_iters

    t1 = sum((theta(1) + theta(2) .* X(k, 2)) - y(k));
    t2 = sum(((theta(1) + theta(2) .* X(k, 2)) - y(k)) .* X(k, 2));

    theta(1) = theta(1) - (alpha / m) * t1;
    theta(2) = theta(2) - (alpha / m) * t2;
 
    J_history(iter) = computeCost(X, y, theta);

end
end
