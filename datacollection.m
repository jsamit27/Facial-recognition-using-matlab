clc    %to obtain a clear screen i.e to clear command window
clear all %clears all data stored to a variable i.e clears workspace
close all %closes all open matlab figure windows
warning off; %to hide any warning messages that might pop up
cao=webcam; %switches on the webcam
faceDetector=vision.CascadeObjectDetector;%each img captured will be detected using viola jones algorithm
c=150;%model will be trained for 150 images for each face captured.
temp=0;%to track no of images captured.
while true
    e=cao.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    else
    es=imcrop(e,bboxes(1,:));%cropping the images
    es=imresize(es,[227 227]);%resizing it to 227*227 which is required by alexnet for training
    filename=strcat(num2str(temp),'.bmp');%the name of the file contains the tempvalue.bmp i.e 0,1,2 etc .bmp file
    imwrite(es,filename);%to write the filename
    temp=temp+1;%value of temp keeps increasing
    imshow(es);
    drawnow;
    end
    else
        imshow(e);
        drawnow;
    end
end

