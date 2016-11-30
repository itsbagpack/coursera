function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_length = size(theta, 1);
cost_factor = lambda/(2*m);
grad_factor = lambda/m;

[main_cost, main_grad] = costFunction(theta, X, y);

reg_term = cost_factor * (sum(theta.^2) - (theta(1)^2));
J = main_cost + reg_term;

grad(1) = main_grad(1);
for i = 2:theta_length
  grad(i) = main_grad(i) + grad_factor*theta(i);
endfor

% =============================================================

end
