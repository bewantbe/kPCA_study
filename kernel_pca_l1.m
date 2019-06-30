% kernel PCA learn
% Case: % explicity kernel

addpath('/home/xyy/matcode/eigen_study/');

% Select kernel function
switch 1
case 1
  ker = @(x, y) (x' * y + 1).^2;
  % this kernel has max dimension 6, since only 6 terms in second order poly.
  % the feature map looks like this
  % 1 + 2 * (x'*y) + (x'*y)^2
  listupper = @(A) triu(A)(triu(ones(length(A)))>0);
  f_feature = @(x) [1; sqrt(2)*x; listupper(x*x'*sqrt(2) - (sqrt(2)-1)*diag(x.*x))];
  % correctness test
	%	x1 = randn(2,1);
	%	y1 = randn(2,1);
	%	ker(x1, y1) - f_feature(x1)' * f_feature(y1)
case 2
  ker = @(x, y) (x' * y + 1).^3;
  % like wise, this dim 10
case 3
  sigma = 1;
  ker = @(x, y) exp(-sum((permute(x, [2,3,1]) - permute(y, [3,2,1])).^2, 3) / (2*sigma^2));
end

% Generate test data
switch 1
case 1
	% circ
	n1 = 100;
	r_d = [
	0.1 + 0.1*rand(1,n1), 1 + 0.2*rand(1,n1), 3 + 0.9*rand(1,n1)
	];
	th_d = 2*pi*rand(1, 3*n1);
	p_d = r_d .* [cos(th_d); sin(th_d)];
	n = size(p_d, 2);
case 2
  n = 300;
  p_d = randn(2, n);
end

% data set scatter plot
figure(3);
plot(p_d(1,:), p_d(2,:), '.');

% Get Gramian matrix and its eigen decomposition
G = ker(p_d, p_d);
% center the data in feature space
%G = G - ones(1,n)/n * G - G * ones(n,1)/n + sum(G(:))/n/n;
[evecs, evals] = eigh((G+G')/2);

% remove zero eigen space
evecs(:, evals < n*1e-12) = [];
evals(evals < n*1e-12) = [];
%evals(evals < 0) = 0;

% normalization in feature space
evecs = evecs .* (1 ./ sqrt(evals'));
evals = evals / n;

% eigenvalue plot
figure(10);
plot(evals, '-o');
ylabel('eigval');
%plot(log10(evals), '-o');
%ylabel('log10(eigval)');

% Get PCA amplitude
pc = evecs' * G;

% PCA plot
figure(12);
plot(pc(1,:), pc(2,:), '.o');
xlabel('pca1');
ylabel('pca2');


if 0

	% check correctness of kernel PCA algorithm
	Y = zeros(6, size(p_d,2));
	for j = 1:size(p_d,2)
		Y(:,j) = f_feature(p_d(:,j));
	end
	[yevec, yeval] = eigh(Y*Y'/n);  % reference answer
	V = Y * evecs;                  % from algo
	% we should have V == yevec in the sense of eigen space
	assert(abs(yevec), abs(V), 1e-10);
	% check principle component
	pc_ref = yevec' * Y;
	assert(abs(pc), abs(pc_ref), 1e-10);

	figure(100);
	plot(yevec(:,1), V(:,1));

	% in genreal impossible
	vs0 = fminsearch(@(v) norm(f_feature(v) - V(:,1)), [0;0]);
end

