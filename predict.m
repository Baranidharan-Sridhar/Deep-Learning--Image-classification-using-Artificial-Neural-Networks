function p = predict(Theta1, Theta2, X)
X = double(X);
m = size(X, 1);
num_labels = size(Theta2, 1);
 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([h1] * Theta2');

[dummy, p] = max(h2, [], 2);

end