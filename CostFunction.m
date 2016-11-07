function [grad J] = CostFunction(neuralNetworkParams, ...
                                   inputLayerSize, ...
                                   hiddenLayerSize, ...
                                   outputLayerSize, ...
                                   X, y, lambda)


  X = double(X);
  
Theta1 = reshape(neuralNetworkParams(1:(hiddenLayerSize+1) * (inputLayerSize+1)), ...
                1+hiddenLayerSize, (inputLayerSize+1));



size(neuralNetworkParams)

Theta2 = reshape(neuralNetworkParams((1 + ((hiddenLayerSize+1) * (inputLayerSize+1))):end), ...
                 outputLayerSize, (1+ hiddenLayerSize));

size(Theta2)
numberOfTrainingExamples = size(X, 1);
 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));




G1 = zeros( size(Theta1) );
G2 = zeros( size(Theta2) );
X = [ones(numberOfTrainingExamples, 1) X];
for i = 1:numberOfTrainingExamples,
	ra1 = X(i, :)';
  
	rz2 = Theta1 * ra1;
	ra2 = sigmoid(rz2);
	

	rz3 = Theta2 * ra2;
	ra3 = sigmoid(rz3);
  
  yVec = (1:outputLayerSize)' == y(i);
	err3 = ra3 - yVec;

	err2 = (Theta2' * err3).* ra2.*(1-ra2);
	G1 = G1 + err2 * ra1';
	G2 = G2 + err3 * ra2';
end


%Theta1_grad = G1 / numberOfTrainingExamples + lambda * [zeros(hiddenLayerSize, 1) Theta1(:, 2:end)] / numberOfTrainingExamples;
%Theta2_grad = G2 / numberOfTrainingExamples + lambda * [zeros(outputLayerSize, 1) Theta2(:, 2:end)] / numberOfTrainingExamples;

Theta1_grad = G1  + (lambda/ numberOfTrainingExamples) * [zeros(size(Theta1, 1),1) Theta1(:, 2:end)] ;
Theta2_grad = G2  + (lambda/ numberOfTrainingExamples) * [zeros(size(Theta2, 1),1) Theta2(:, 2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];
fprintf("dammit")
size(grad)
end