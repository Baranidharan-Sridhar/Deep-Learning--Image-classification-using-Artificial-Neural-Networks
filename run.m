clear ; close all; clc

inputLayerSize  = 240;  
hiddenLayerSize = 4; 
outputLayerSize= 2; 
intialEpsilon = 0.15; 
x= zeros(39,240);
lambda=2;

filelist = dir ('images/Training/*.pgm');
array = cell(1, length(filelist)-2);


for i=1:numel(filelist)
    array{i}= imread(['images/Training/' filelist(i,1).name]);
    o= reshape(array{i},1,240);
    x=[x;o]; 
    
endfor
X=x(40:78,:);
y=zeros(20,1);
y=[y;ones(19,1)];

initialTheta1= rand(1 + hiddenLayerSize, inputLayerSize) * 2 * intialEpsilon - intialEpsilon;
initialTheta2= rand( 1+outputLayerSize, 1 +hiddenLayerSize) * 2 * intialEpsilon - intialEpsilon;
size(initialTheta1)
size(initialTheta2)

neuralNetworkParams = [initialTheta1(:) ; initialTheta2(:)];
size(neuralNetworkParams)
[nNetworkParams, cost]  = CostFunction(neuralNetworkParams, inputLayerSize, hiddenLayerSize, outputLayerSize, X, y, lambda);

Theta1 = reshape(nNetworkParams(1:(hiddenLayerSize+1) * (inputLayerSize+1)), ...
                1+hiddenLayerSize, (inputLayerSize+1));



Theta2 = reshape(nNetworkParams((1 + ((hiddenLayerSize+1) * (inputLayerSize+1))):end), ...
                 outputLayerSize, (1+ hiddenLayerSize));

 

test_filelist = dir ('images/Test/*.pgm');
test_array = cell(1, length(filelist)-2);

test_x= zeros(20,240);
for i=1:numel(test_filelist)
    test_array{i}= imread(['images/Test/' test_filelist(i,1).name]);
    test_o= reshape(test_array{i},1,240);
    test_x=[test_x;test_o]; 
    test_y=zeros(10,1);
test_y=[test_y;ones(10,1)];
    
endfor
test_X=test_x(21:40,:);
pred = predict(Theta1, Theta2, test_X);

for int=1: numel(pred)
    if (pred(int,1)== 2)
      fprintf("The prediction for %s is Tree \n",test_filelist(int,1).name)
     else
       fprintf("The prediction for %s is ball\n",test_filelist(int,1).name)
     endif
endfor     
