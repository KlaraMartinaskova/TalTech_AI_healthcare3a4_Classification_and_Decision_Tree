% Author: Klara Martinaskova
% AI in Healthcare 2023
% MATLAB Assignments 3-4

% Multiclass classification
%% 
clear all
clc
%% Load data
val = load('AccSignalVal.mat'); % data for validation
train = load('AccSignal.mat'); % data for training

% whole signal in three axis
% all samples in one long vectors
accX = train.accX;
accY = train.accY;
accZ = train.accZ;

% Acceleration segments for each different axis
% Divides long vectors to segments
% Columns (1. column = 1-125 samples, 2. column = 126 - 251 samples)
accSegX = train.accSegX;
accSegY = train.accSegY;
accSegZ = train.accSegZ;

% Label
labelVal = train.accLabel;
% categorized for each part (each column)
% 1 = Static (either lying, standing or sitting)
% 2 = Walking
% 3 = Running
% 4 = Other (this includes various upper-body movement)
%% Plot data

% Plot 3 signals to one figure
figure (1)
hold on
plot(accX)
plot(accY)
plot(accZ)
title('Acceleration')
legend('Axis X', 'Axis Y', 'Axis Z')

% Plot lables
figure(2)
plot(labelVal)
title('Labels')
%% Extract features
% For each segment calculate mean and std for each axis
% Matrix 660x6 (but the first number could be different)
% Structure of one row in matrix: meanX meandY meanZ stdX stdY stdZ

featuresMatrix = zeros(length(accSegX), 6); % helper variable for set results

% Function for calculation mean and std for each axis:
[meanX, stdX] = MeanAndStd(accSegX); 
[meanY, stdY] = MeanAndStd(accSegY);
[meanZ, stdZ] = MeanAndStd(accSegZ);

% Set results to matrix according to structure:
% meanX meandY meanZ stdX stdY stdZ
featuresMatrix(:,1) = meanX;
featuresMatrix(:,2) = meanY;
featuresMatrix(:,3) = meanZ;
featuresMatrix(:,4) = stdX;
featuresMatrix(:,5) = stdY;
featuresMatrix(:,6) = stdZ;

%% Training a decision tree based classifier
% Decision tree classifier is based on the training set.

classifierTree = fitctree(featuresMatrix, labelVal);

%% Graphically represent
% Observe the tree: is it possible to understand how the classifier works based on the image?
view(classifierTree,'mode','graph')

%% Evaluate the classifier
% Separate data for the training set and evaluation set.

% Data for validation:
valSegX = val.accSegXVal;
valSegY = val.accSegYVal;
valSegZ = val.accSegZVal;

labelVal = val.accLabelVal; % labels

% Extract features (same code as for training data):
featuresMatrixVal = zeros(length(valSegX), 6);

[meanX, stdX] = MeanAndStd(valSegX);
[meanY, stdY] = MeanAndStd(valSegY);
[meanZ, stdZ] = MeanAndStd(valSegZ);

featuresMatrixVal(:,1) = meanX;
featuresMatrixVal(:,2) = meanY;
featuresMatrixVal(:,3) = meanZ;
featuresMatrixVal(:,4) = stdX;
featuresMatrixVal(:,5) = stdY;
featuresMatrixVal(:,6) = stdZ;

%% Classify the evaluation set features
predictedLabelVal = predict(classifierTree, featuresMatrixVal); 

%% Classification results

% Confusion Matrix
C = confusionmat(labelVal, predictedLabelVal);
disp(C)

figure(3)
confusionchart(C) % for displaying results

% Find the TP, TN, FP, FN

labels = unique(labelVal); % get unique labels in data

% Initialize empty arrays to store the results for each label
TP = zeros(size(labels));
TN = zeros(size(labels));
FN = zeros(size(labels));
FP = zeros(size(labels));
accuracy = zeros(size(labels));

for i = 1:length(labels)
    currentLabel = labels(i);
    for segment = 1:length(predictedLabelVal)
        if predictedLabelVal(segment) == currentLabel & labelVal(segment) == currentLabel % TP
            TP(i) = TP(i) + 1;
        elseif predictedLabelVal(segment) ~= currentLabel & labelVal(segment) ~= currentLabel % TN
            TN(i) = TN(i) + 1;
        elseif predictedLabelVal(segment) ~= currentLabel & labelVal(segment) == currentLabel % FN
            FN(i) = FN(i) + 1;
        elseif predictedLabelVal(segment) == currentLabel & labelVal(segment) ~= currentLabel % FP
            FP(i) = FP(i) + 1;          
        end      
    end
    accuracy(i) = (TP(i) + TN(i)) / (TP(i) + TN(i) + FP(i) + FN(i));
end

% Display the results for each label
for i = 1:length(labels)
    fprintf('Label %d: TP=%d, TN=%d, FP=%d, FN=%d\n', ...
        labels(i), TP(i), TN(i), FP(i), FN(i));
end


% Calculation the evaluation metrics for each activity type
TPR = TP ./ (TP + FN);
TNR = TN ./ (TN + FP);
FPR = FP ./ (FP + TN);
FNR = FN ./ (FN + TP);

% Print the evaluation metrics for each activity type
for i = 1:length(labels)
    fprintf('Activity type %d:\n', labels(i));
    fprintf('True Positive Rate (Sensitivity/Recall): %f\n', TPR(i));
    fprintf('True Negative Rate (Specifity/Selectivity): %f\n', TNR(i));
    fprintf('False Positive Rate (False Alarm Rate/Type I Error): %f\n', FPR(i));
    fprintf('False Negative Rate (Miss Rate/Type II Error): %f\n', FNR(i));
    fprintf('Accuracy: %f\n\n', accuracy(i));
end

% Create a table to display the evaluation metrics for each label
metricsTable = table(TPR, TNR, FPR, FNR, accuracy, 'RowNames', cellstr(num2str(labels)));
disp(metricsTable); % display the table

%% Answering the questions in enclosed PDF file