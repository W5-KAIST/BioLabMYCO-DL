function scores=cnn_vgg_function(net,name)
%CNN_VGG_FACES  Demonstrates how to use VGG-Face
%net = load('data/models/vgg-face.mat') ;
im=imread(name);
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;

res = vl_simplenn(net, im_) ;

% just before softmax
scores = squeeze(gather(res(end-2).x)) ;

