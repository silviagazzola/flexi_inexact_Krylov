function [X, Enrm, Rnrm, V, U, Y, M, T] = APD(A, b, MaxIter, xexact, x0, tol, rest, pnorm)
% APD inexact/flexible solver for p-norm data fitting problems
%
% [X, Enrm, Rnrm, V, U, Y, M, T] = DAP(A, b, MaxIter, xexact, x0, tol)
% [X, Enrm, Rnrm, V, U, Y, M, T] = DAP(A, b, MaxIter, xexact, x0, tol, rest)
% [X, Enrm, Rnrm, V, U, Y, M, T] = DAP(A, b, MaxIter, xexact, x0, tol, rest, pnorm)
%
% DAP is an iterative regularization method used for 
% solving p-norm data fitting problems. DAP is based on an inexact/flexible
% Golub-Kahan factorization, which allows weight updates along the iterations.
%
%
% Inputs:
%  A : either (a) a full or sparse matrix
%             (b) a matrix object that performs the matrix*vector operation
%             (c) user-defined function handle
%  b : right-hand side vector
%  MaxIter : the maximum number of iterations
%      [ positive integer | vector of positive components ]
% xexact   : true solution; allows us to return error norms with
%            respect to xexact at each iteration.
%    x0    : initial guess for the iterations
%     tol  : tolerance on the residual for restarting
%    rest  : restarting strategy
%            [ 'off' | 'weights' | 'res' ]
%   pnorm  : value of p for the p-norm data fitting problem
%
% Outputs:
%   X   : computed solutions, stored column-wise at each iterations listed in K)
% Enrm  : relative error norms at each iteration (requires x_true)
% Rnrm  : relative residual norms at each iteration

% Solver described in: `New flexible and inexact Golub-Kahan algorithms for
% inverse problems', arXiv:2510.18865

% Silvia Gazzola, Universita' di Pisa
% Malena Sabate Landman, University of Bath
% March, 2026

if nargin == 6
    rest = 'on';
    rest_type = 'weights';
    pnorm = 1;
elseif nargin == 7 || nargin == 8
    if strcmpi(rest, 'res')
     rest_type = 'res';
     rest = 'on';
    elseif strcmpi(rest, 'weights')
     rest_type = 'weights';
     rest = 'on';
    elseif strcmpi(rest, 'off')
     rest_type = 'off';
    end
    if nargin == 7, pnorm = 1; end
end

p = pnorm;
tolX = 1e-3;

nrmex = norm(xexact(:));
nrmb = norm(b(:));

m = size(b,1);
n = size(x0,1);
X = zeros(n,MaxIter);
W_store = zeros(m, MaxIter+1);
Enrm = zeros(MaxIter, 1);
Rnrm = zeros(MaxIter, 1);

r = b(:) - A*x0;
initial_r = r;
u = r;
beta = norm(u(:)); 
rhs = zeros(MaxIter+1,1);
rhs(1) = beta;
U(:,1) = u/beta;

w = r;
w = (w.^2 + tolX.^2).^((p-2)/2);
W_store(:,1)=w;

uiRkr0 = zeros(MaxIter,1);
uiRkr0(1) = U(:,1)'*(w.*initial_r);

y = w.*U(:,1);
Y(:,1) = y;
v = Atransp_times_vec(A, Y(:,1)); v = v(:);
T(1,1) = norm(v);
v = v / T(1,1);
V(:,1) = v;
M = zeros(MaxIter+1,MaxIter);
for k=1:MaxIter
    
    u = A_times_vec(A, v); u = u(:);
    for i = 1:k
        M(i,k) = U(:,i)'*u;
        u = u - M(i,k)*U(:,i);
    end
    M(k+1,k) = norm(u);
    u = u / M(k+1,k);
    U(:,k+1) = u;
    uiRkr0(k+1) = u'*(w.*initial_r);

    y = w.*u;
    Y(:,k+1) = y;
    
    v = Atransp_times_vec(A, Y(:,k+1)); v = v(:);
    for i = 1:k
        T(i,k+1)=V(:,i)'*v;
        v = v - T(i,k+1)*V(:,i);
    end
    T(k+1, k+1) = norm(v);
    v = v / T(k+1, k+1);
    V(:, k+1) = v;
    
    rhsproj = T(1,1)*rhs(1:k)+ (M(1:k+1,1:k))'*uiRkr0(1:k+1);
    Aproj = (T(1:k,:)*M(1:k+1,1:k))' + T(1:k,:)*M(1:k+1,1:k);
    s = Aproj\rhsproj;
    x = x0 + V(:,1:k)*s;
    X(:,k) = x;
    Enrm(k) = norm(x - xexact)/nrmex;

    rproj = rhs(1:k+1) - M(1:k+1,1:k)*s;
    Rnrm(k) = norm(rproj)/nrmb;
    r = U*(rproj);
    w = (r.^2 + tolX.^2).^((p-2)/2);
    W_store(:,k+1)=w;
    nErr_curr = zeros(k+1,1);
    for i = 1:k+1
        nErr_curr(i) = max(abs(W_store(:,i) - w));
    end
    if strcmpi(rest, 'on') 
        if (strcmpi(rest_type, 'weights') && ~isempty(find(nErr_curr(1:end-1) - nErr_curr(2:end)<0,1)))...
                || (strcmpi(rest_type, 'res') && (norm(r-initial_r))/norm(r)> tol)
                X = X(:,1:k-1);
                Enrm = Enrm(1:k-1);
                Rnrm = Rnrm(1:k-1);
                break
        end
    end
end