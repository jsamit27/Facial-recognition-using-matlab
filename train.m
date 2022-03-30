clc
clear all
close all
g=alexnet;
layers=g.Layers;
layers(23)=fullyConnectedLayer(3);
layers(25)=classificationLayer;%A layer in a deep learning model is a structure or network topology in the architecture
                               %of the model, which take information from the previous layers and then pass information to the next layer.
allImages=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames');
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
myNet=trainNetwork(allImages,layers,opts);
save myNet1;
%One Epoch is when an ENTIRE dataset is 
%passed forward and backward through the neural network only ONCE. Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.
%Iteration is one time processing for forward and backward for a batch of images (say one batch is defined as 16, then 16 images are processed in one iteration)
%When the batch size is more than one sample and less than the size of the training dataset, the learning algorithm is called mini-batch gradient descent.
%he learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.
%The learning rate controls how quickly the model is adapted to the problem.
