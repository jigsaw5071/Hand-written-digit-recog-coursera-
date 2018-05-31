function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));




H = sigmoid(X * theta);
J = (-1.0 * sum(y' * log(H) + (1 - y)' * log(1 - H)))/m;
J = J + (lambda / (2.0 * m)) * (sum(theta(2 : end) .^2));

grad = (X' * (H - y))/m;
grad(2 : end) = grad(2 : end) + (lambda / m) * theta(2 : end);






% =============================================================

grad = grad(:);

end
