function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
Y = zeros(size(y,1),num_labels);
for i = 1:length(y)
Y(i,y(i)) = 1;
end
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1),X];
z1 = X*Theta1';
a1 = sigmoid(z1);
a1 = [ones(size(a1,1),1),a1];
z2 = a1*Theta2';
h = sigmoid(z2);
J = -1/m*(Y.*log(h)+(1-Y).*log(1-h));
J = sum(J,1);
J = sum(J,2);
J_reg = lambda/(2*m)*((nn_params'*nn_params)-(Theta1(:,1)'*Theta1(:,1)+Theta2(:,1)'*Theta2(:,1)));
J = J+J_reg;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta_3 = zeros(size(Y));
delta_2 = zeros(size(a1));
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for t = 1:m
    
    delta_3(t,:) = h(t,:)-Y(t,:);
    delta_2(t,:) = (delta_3(t,:)*Theta2).*[1,sigmoidGradient(z1(t,:))];
    D2 = D2+delta_3(t,:)'*a1(t,:);
    D1 = D1+delta_2(t,2:end)'*X(t,:);    
end

Theta1_grad = 1/m*D1;
Theta2_grad = 1/m*D2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_reg =lambda/m*Theta1(:,2:end)+Theta1_grad(:,2:end);
Theta2_reg =lambda/m*Theta2(:,2:end)+Theta2_grad(:,2:end);

Theta1_grad = [Theta1_grad(:,1),Theta1_reg];

Theta2_grad = [Theta2_grad(:,1),Theta2_reg];







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
