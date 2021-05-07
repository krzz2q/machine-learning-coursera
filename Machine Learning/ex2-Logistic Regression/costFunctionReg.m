function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
 %DIMENSIONS: 
  %   theta = (n+1) x 1
  %   X     = m x (n+1)
  %   y     = m x 1
  %   grad  = (n+1) x 1
  %   J     = Scalar
m = length(y); % number of training examples
grad = zeros(size(theta));

% You need to return the following variables correctly 
J = sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))/m+lambda*sum(theta(2:end).*theta(2:end))/(2*m);
grad(1) = (1/m)* (X(:,1)'*(sigmoid(X*theta)-y));                                  % 1 x 1
grad(2:end) = (1/m)* (X(:,2:end)'*(sigmoid(X*theta)-y))+(lambda/m)*theta(2:end);  % n x 1
%grad(2:end) = (1/m)* (X(:,2:end)'*(h_x-y))+(lambda/m)*theta(2:end); 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
