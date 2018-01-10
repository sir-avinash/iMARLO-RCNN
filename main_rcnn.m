%% Object Detection Notes: https://www.mathworks.com/help/vision/examples/object-detection-using-deep-learning.html#d119e1152
%% Annotation: https://www.mathworks.com/help/vision/ref/imagelabeler-app.html


%% Use imageLabeler to annotate pics
labelData = 0;

if labelData
    if strcmp(version,'9.2.0.556344 (R2017a)')
        trainingImageLabeler
    else    
        imageLabeler
    end
end
    
clear all
%% Load Data
load('step1Data.mat', 'step1Data')

%% Point to Images Directory
imDir =  fullfile(pwd,'sample_data_steplen','multitexture_wshadow');
addpath(imDir);

%% Training Options
options = trainingOptions('sgdm', ...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs',  10);

% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
doTraining = false;

if doTraining
    % Train a network.
    %cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
    error('Network Learning not setup yet')
    %% You can transfer network and weight from Keras! Yay!
else
    % Load pre-trained detector for the example.
    load('rcnnStopSigns.mat','cifar10Net')
end

%% Traning - use either 'trainRCNNObjectDetector' or ''
rcnn = trainFasterRCNNObjectDetector(step1Data, cifar10Net, options, 'NegativeOverlapRange', [0 0.3]);

%% Test
test_img = fullfile(pwd,'sample_data_steplen','multitexture_wshadow','image14.png');
img = imread(test_img);

[bbox, score, label] = detect(rcnn, img) %, 'MiniBatchSize', 32);

%% Visualize result
% [score, idx] = max(score);

% bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)

%% Remove directory from Path



%%% Stop Sign Detection Code
% load('rcnnStopSigns.mat', 'stopSigns', 'layers')
% layers
% imDir = fullfile(matlabroot, 'toolbox', 'vision', 'visiondata',...
% 'stopSignImages');
% addpath(imDir);
% options = trainingOptions('sgdm', ...
% 'MiniBatchSize', 32, ...
% 'InitialLearnRate', 1e-6, ...
% 'MaxEpochs', 10);
% rcnn = trainRCNNObjectDetector(stopSigns, layers, options, 'NegativeOverlapRange', [0 0.3]);
% img = imread('stopSignTest.jpg');
% imshow(img)
% [bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);
% [score, idx] = max(score);
% bbox = bbox(idx, :);
% annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
% detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);
% figure
% imshow(detectedImg)
