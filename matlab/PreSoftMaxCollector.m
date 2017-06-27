clear
addpath matlab
run vl_setupnn
tic
posPath = './tbtest/pos/';
negPath = './tbtest/neg/';
cnnNet = 'imagenet-vgg-verydeep-16';
projName = 'tbtest';
%cnnNet = 'imagenet-vgg-verydeep-16';

disp('Loading Pre-trained Net...')
net = load([cnnNet '.mat']) ;

disp('Loading negative file list...')
negList = dir(fullfile(negPath,'*.png'));
negList = struct2dataset(negList);
negList = negList.name;

disp('Loading positive file list...')
posList = dir(fullfile(posPath,'*.png'));
posList= struct2dataset(posList);
posList = posList.name;

negData=transpose(cnn_vgg_function(net,[negPath negList{1}]));
disp(['processing negative ' num2str(1) ' / ' num2str(length(negList))]);
disp(negList{2});
for j=2:(length(negList))
    disp(['processing negative ' num2str(j) ' / ' num2str(length(negList))]);
    
    negData=[negData;transpose(cnn_vgg_function(net,[negPath negList{j}]))];
end
disp('job done');

posData=transpose(cnn_vgg_function(net,[posPath posList{1}]));
disp(['processing positive ' num2str(1) ' / ' num2str(length(posList))]);
for j=2:(length(posList))
    disp(['processing positive ' num2str(j) ' / ' num2str(length(posList))]);
    posData=[posData;transpose(cnn_vgg_function(net,[posPath posList{j}]))];
end
disp('job done');
save([projName '-' cnnNet '-presoftmax.mat'],'negData','posData','posList', 'negList')
disp(['Data saved at ' [projName '-' cnnNet '-presoftmax.mat']])
toc