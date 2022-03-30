clc;close;clear
c=webcam;%camera being switched on
load myNet1;
faceDetector=vision.CascadeObjectDetector;%viola jones algorithm
while true
    e=c.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)%face detected or not
     es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    label=classify(myNet,es);
    image(e);
    title(char(label));%recognizing and providing title
    drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end