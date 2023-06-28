function [meanList, stdList] = MeanAndStd(axis)

% This function deals with calculations features mean and std.
% It calculates features for each segments for one given axis
% Input: Axis (devides to segments by columns)
% Output: Mean and Std for each segment in given axis

[~, numSegments]=size(axis);
meanList = zeros(numSegments,1);
stdList = zeros(numSegments,1);


for i = 1:numSegments
    currentSegment = axis(:,i);
    meanList(i)= mean(abs(currentSegment));
    stdList(i) = std(currentSegment);
end
end


