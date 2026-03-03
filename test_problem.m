% Requires IR Tools: https://github.com/jnagy1/IRtools

% Solver described in: `New flexible and inexact Golub-Kahan algorithms for
% inverse problems', arXiv:2510.18865

% Silvia Gazzola, Universita' di Pisa
% Malena Sabate Landman, University of Bath
% March, 2026


clear, clc

n = 256;
rng(0) % reproducibility

% generate the blurring
PSF = psfNSGauss([128, 128], 4, 10, 2);
optbl.CommitCrime = 'on';
optbl.PSF = PSF;
optbl.trueImage = 'hst';
[A, bexact, xexact, ProbInfo] = PRblur(n, optbl);

% rescale
scale = max(bexact);
xexact = xexact/scale;
bexact = bexact/scale;
nrmex = norm(xexact(:));
% add noise
b = imnoise(bexact, 'salt & pepper', 0.1);

% solver parameters
tolX = 1e-3;
MaxIter = 100;
tol = 1e-1;

% DAP method
Enrm_DAP = []; 
ind_DAP = [];
xtemp = zeros(n^2,1);
Xall_DAP = zeros(n^2, MaxIter);
Xrs_DAP = zeros(n^2, MaxIter);
for i = 1:MaxIter
    [Xtemp, Enrmtemp, Rnrmtemp] = DAP(A, b, MaxIter, xexact, xtemp, tol, 'weights', 1);
    Enrm_DAP = [Enrm_DAP; Enrmtemp(1:end-1)];
    newbit = [length(Enrm_DAP), length(Enrmtemp(1:end-1))];
    ind_DAP = [ind_DAP; newbit];
    if size(Enrm_DAP,1)>100
        break
    end
    xtemp = Xtemp(:,end);
    Xrs_DAP = [Xrs_DAP, xtemp];
    Xall_DAP = [Xall_DAP, Xtemp];
end

% ADP method
Enrm_APD = []; 
ind_APD = [];
xtemp = zeros(n^2,1);
Xall_APD = zeros(n^2, MaxIter);
Xrs_APD = zeros(n^2, MaxIter);
for i = 1:MaxIter
    [Xtemp, Enrmtemp, Rnrmtemp] = APD(A, b, MaxIter, xexact, xtemp, tol, 'weights', 1);
    Enrm_APD = [Enrm_APD; Enrmtemp(1:end-1)];
    newbit = [length(Enrm_APD), length(Enrmtemp(1:end-1))];
    ind_APD = [ind_APD; newbit];
    if size(Enrm_APD,1)>100
        break
    end
    xtemp = Xtemp(:,end);
    Xrs_APD = [Xrs_APD, xtemp];
    Xall_APD = [Xall_APD, Xtemp];
end

% plot the errors
figure, semilogy(Enrm_DAP, 'LineWidth', 2)
hold on 
semilogy(Enrm_APD, 'LineWidth', 2)
legend('DAP', 'APD')
xlabel('iterations')
ylabel('RRE')

function PSF = psfNSGauss(dim, sigma1, sigma2, rho)
%
%        PSF = psfGauss(dim, sigma);
%
%  This function constructs a gaussian blur PSF. 
%
%  Input: 
%    dim  -  desired dimension of the pointspread function
%            e.g., PSF = psfGauss([60,60]) creates a 60-by-60 
%            Gaussian point spread function.
%
%  Optional input parameters:
%    sigma  -  variance of the gaussian
%              Default is sigma = 2.0.
%

if ( nargin == 1 )
  sigma1 = 2; sigma2 = 2; rho = 0;
elseif (nargin == 2)
  sigma2 = sigma1; rho = 0;
elseif (nargin == 3)
  rho = 0;
end

l = length(dim);

switch l
case 1
  x = -fix(dim(1)/2):ceil(dim(1)/2)-1;
  y = 0;
case 2
  x = -fix(dim(1)/2):ceil(dim(1)/2)-1;
  y = -fix(dim(2)/2):ceil(dim(2)/2)-1;
case 3
  x = -fix(dim(1)/2):ceil(dim(1)/2)-1;
  y = -fix(dim(2)/2):ceil(dim(2)/2)-1;
otherwise
  error('illegal PSF dimension')
end

z = 0;
[X,Y] = meshgrid(x,y,z);

PSF = exp( -1/(2*(sigma1^2*sigma2^2 - rho^4))*( sigma2^2*X.^2 - 2*rho^2*X.*Y + sigma1^2*Y.^2 ) );
end