%% data prepossing for training set 
imagefiles = dir('./trainset/*/*.jpg');
nfiles = length(imagefiles);
images = zeros(2500,240);
for ii = 1:nfiles
    currentfilename = imagefiles(ii).name;
    thisFileName = fullfile(imagefiles(ii).folder, imagefiles(ii).name);
    currentimage = imread(thisFileName);
    reshapecurrentimage = reshape(currentimage,[2500,1]);
    images(:,ii) = reshapecurrentimage;   
end

%% data prepossing for testing set 
imagefiles = dir('./testset/*/*.jpg');
nfiles = length(imagefiles);
images_test = zeros(2500,60);
for ii = 1:nfiles
    currentfilename = imagefiles(ii).name;
    thisFileName = fullfile(imagefiles(ii).folder, imagefiles(ii).name);
    currentimage = imread(thisFileName);
    reshapecurrentimage = reshape(currentimage,[2500,1]);
    images_test(:,ii) = reshapecurrentimage;   
end


%% Principle component for 5a

meantotal = mean(images'); 
covtotal = cov(images');
[U S V] = svd(covtotal);
phi = U(:,1:15);
 figure
for i = 1:16 
     subplot(4,4,i);
     imagesc(reshape(U(:,i),[50,50]));
     colormap(gray(255));
end 

%%this part is for 5c, pca for training and testing 

pca_training_images = phi'*(images-meantotal');
meantotal_test = mean(images_test'); 
pca_testing_images =  phi'*(images_test-meantotal_test');
pca_gaussian_mean = zeros(15,6);
pca_gaussian_cov  =zeros(15,15,6);
for i =0:5
    pca_face = pca_training_images(:,40*i+1:40*(i+1));
    pca_gaussian_mean(:,i+1) = mean(pca_face');
    tmpcov  = cov(pca_face');
    tmpcov = (tmpcov+tmpcov')/2;
    pca_gaussian_cov(:,:,i+1) = tmpcov;
end 
%%Changing the order of the pca_testing_images 
pca_testing_images = pca_testing_images(:,[21:60 1:20]);
pca_correct = zeros(1,6);
for i = 1:60
    tmpmax = mvnpdf(pca_testing_images(:,i),pca_gaussian_mean(:,ceil(i/10)),pca_gaussian_cov(:,:,ceil(i/10)));
    max1 =-100;
    for j = 1:6
         max1 = max(mvnpdf(pca_testing_images(:,i),pca_gaussian_mean(:,j),pca_gaussian_cov(:,:,j)),max1);
    end
     if tmpmax == max1
        pca_correct(1,ceil(i/10)) = pca_correct(1,ceil(i/10))+1;
     end
end

pca_error = (10-pca_correct)/10
mean_pca_error = mean(pca_error)



%% LDA for 5b
%  this is for the first two subset

figure 
count =1;
for i = 0:4
    for j = i+1:5
        images1 = images(:,40*i+1:40*(i+1));
        cov1 = cov(images1');
        mean1 = mean(images1');
        images2 = images(:,40*j+1:40*(j+1));
        cov2 = cov(images2');
        mean2 = mean(images2');
        w = inv(cov1+cov2+eye(2500))*(mean2'-mean1');
        subplot(4,4,count);
        imagesc(reshape(w,[50,50]));
        colormap(gray(255));
        count=count+1;
    end 
end


%% LDA for 5d
lda_w = zeros(2500,15);
count =1;
for i = 0:4
    for j = i+1:5
        images1 = images(:,40*i+1:40*(i+1));
        cov1 = cov(images1');
        mean1 = mean(images1');
        images2 = images(:,40*j+1:40*(j+1));
        cov2 = cov(images2');
        mean2 = mean(images2');
        w = inv(cov1+cov2+eye(2500))*(mean2'-mean1');
        lda_w(:,count) = w;
        count=count+1;
    end 
end


lda_training_images = lda_w'*images;
% meantotal_test = mean(images_test'); 
lda_testing_images =  lda_w'*images_test;
lda_gaussian_mean = zeros(15,6);
lda_gaussian_cov  =zeros(15,15,6);
for i =0:5
    lda_face = lda_training_images(:,40*i+1:40*(i+1));
    lda_gaussian_mean(:,i+1) =mean(lda_face');
    tmpcov  = cov(lda_face');
    tmpcov = (tmpcov+tmpcov')/2+1000000000*eye(15);
    lda_gaussian_cov(:,:,i+1) = tmpcov;
end 
%%Changing the order of the pca_testing_images 
lda_testing_images = lda_testing_images(:,[21:60 1:20]);
lda_correct = zeros(1,6);
for i = 1:60
    tmpmax = mvnpdf(lda_testing_images(:,i),lda_gaussian_mean(:,ceil(i/10)),lda_gaussian_cov(:,:,ceil(i/10)));
    max1 =-100;
    for j = 1:6
         max1 = max(mvnpdf(lda_testing_images(:,i),lda_gaussian_mean(:,j),lda_gaussian_cov(:,:,j)),max1);
    end
     if tmpmax == max1
        lda_correct(1,ceil(i/10)) = lda_correct(1,ceil(i/10))+1;
     end
end
lda_error = (10-lda_correct)/10;
mean_lda_error = mean(lda_error);


%%5e PCA + LDA
meantotal = mean(images'); 
covtotal = cov(images');
[U S V] = svd(covtotal);
phi = U(:,1:30);
pca_training_images = phi'*(images-meantotal')
meantotal_test = mean(images_test'); 
pca_testing_images =  phi'*(images_test-meantotal_test');
lda_w = zeros(30,15);
count =1;
for i = 0:4
    for j = i+1:5
        images1 = pca_training_images(:,40*i+1:40*(i+1));
        cov1 = cov(images1');
        mean1 = mean(images1');
        images2 = pca_training_images(:,40*j+1:40*(j+1));
        cov2 = cov(images2');
        mean2 = mean(images2');
        w = inv(cov1+cov2+eye(30))*(mean2'-mean1');
        lda_w(:,count) = w;
        count=count+1;
    end 
end
lda_training_images = lda_w'*(pca_training_images-mean(pca_training_images));
% meantotal_test = mean(images_test'); 
lda_testing_images =  lda_w'*(pca_testing_images-mean(pca_testing_images));
lda_gaussian_mean = zeros(15,6);
lda_gaussian_cov  =zeros(15,15,6);
for i =0:5
    lda_face = lda_training_images(:,40*i+1:40*(i+1));
    lda_gaussian_mean(:,i+1) =mean(lda_face');
    tmpcov  = cov(lda_face');
     tmpcov = (tmpcov+tmpcov')/2;
    lda_gaussian_cov(:,:,i+1) = tmpcov;
end 
%%Changing the order of the pca_testing_images 
lda_testing_images = lda_testing_images(:,[21:60 1:20]);
lda_correct = zeros(1,6);
for i = 1:60
    tmpmax = mvnpdf(lda_testing_images(:,i),lda_gaussian_mean(:,ceil(i/10)),lda_gaussian_cov(:,:,ceil(i/10)));
    max1 =-100;
    for j = 1:6
         max1 = max(mvnpdf(lda_testing_images(:,i),lda_gaussian_mean(:,j),lda_gaussian_cov(:,:,j)),max1);
    end
     if tmpmax == max1
        lda_correct(1,ceil(i/10)) = lda_correct(1,ceil(i/10))+1;
     end
end
pcalda_error = (10-lda_correct)/10;
mean_pcalda_error = mean(lda_error);
