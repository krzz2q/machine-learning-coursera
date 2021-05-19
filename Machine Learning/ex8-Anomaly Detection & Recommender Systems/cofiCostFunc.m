function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
yp=X*Theta';
J=1/2*sum(sum(R.*(yp-Y).^2))+lambda/2*(sum(sum(Theta.^2))+sum(sum(X.^2)));

for i=1:num_movies%更新feature
    idx=find(R(i,:)==1);%已对i电影评分的用户j
    temp_theta=Theta(idx,:);%已评分用户的参数
    temp_Y=Y(i,idx);%已评分用户对电影i的评分
    X_grad(i,:)= (X(i,:)*temp_theta'-temp_Y)*temp_theta+lambda*X(i,:);
end
    
    

for j=1:num_users%更新theta
    idx=find(R(:,j)==1);%用户j评分的电影i
    temp_X=X(idx,:);%已评分电影的特征
    temp_Y=Y(idx,j);%j用户对已评分电影的评分
    Theta_grad(j,:)=(temp_X*Theta(j,:)'-temp_Y)'*temp_X+lambda*Theta(j,:);
end












% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
