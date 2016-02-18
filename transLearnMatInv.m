function [ M ] = transLearnMatInv( X, Z, lambda, regType)

% min_{M} sum_{i = 1,...,n} |Mx_i - z_i|^2 + lambda * |M|_some_norm
% X = [x_1'; x_2';...; x_n'];
% Z = [z_1'; z_2';...; z_n'];
% lambda is the parameter controls regularization
% if regType == 1, no regularization is used , if regType == 2, use 
% frobenius norm

[~,dx] = size(X);
[~,dz] = size(Z);

I = eye(dx);

%% compute gradient
if regType == 1
    M = (X'*X)\(X'*Z);
    % solve for grad = (M*X'-Z')*X = 0, that is no norm;
elseif regType == 2
    M = (X'*X + lambda*I)\(X'*Z);
    % solve for grad = (M*X'-Z')*X + lambda*M = 0; %% frobenius norm |M|^2
end

M = M';

end

