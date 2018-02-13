% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images_train = loadMNISTImages('train-images-idx3-ubyte')';
labels_train = loadMNISTLabels('train-labels-idx1-ubyte');
y_train = labels_train;
labels_train = onehot(labels_train,10);

for it=1:60000
    m = mean(images_train(it,:));
    images_train(it,:) = images_train(it,:)-m;
end

images_test = loadMNISTImages('t10k-images-idx3-ubyte')';
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
y_test = labels_test;
labels_test = onehot(labels_test,10);


for it=1:10000
    m = mean(images_test(it,:));
    images_test(it,:) = images_test(it,:)-m;
end

ntrain = 50000;
nval = 10000;
ntest = 10000;

train_x = [ones(ntrain,1),images_train(1:ntrain,:)];
train_y = labels_train(1:ntrain,:)';

val_x = [ones(nval,1),images_train(ntrain+1:ntrain+nval,:)];
val_y = labels_train(ntrain+1:ntrain+nval,:)';

test_x = [ones(ntest,1),images_test(1:ntest,:)];
test_y = labels_test(1:ntest,:)';

% Setting hyperparameters
n_iters = 200;
nh = 400;
ni = 784;
no = 10;
eta0 = 0.3;
eta = eta0;
T = 10000;
mom = 0.7;
trainsize = 125;

% Initialization considering fan-in
w1 = normrnd(0,1/sqrt(ni),[ni+1,nh]);
w1(end,:) = normrnd(0,1,[1,nh]);
w2 = normrnd(0,1/sqrt(nh),[nh+1,no]);
w2(end,:) = normrnd(0,1,[1,no]);

% Initialization without fan-in
% w1 = normrnd(0,1,[ni+1,nh]);
% w1(end,:) = normrnd(0,1,[1,nh]);
% w2 = normrnd(0,1,[nh+1,no]);
% w2(end,:) = normrnd(0,1,[1,no]);

% Shuffling data
r = randi([1 ntrain],1,ntrain);
train_x(1:ntrain,:) = train_x(r,:);
train_y(:,1:ntrain) = train_y(:,r);

tr_ac = zeros([1,n_iters]);
val_ac = zeros([1,n_iters]);
te_ac = zeros([1,n_iters]);
time = 0.0;
for i=1:n_iters
    tic;
    % Running mini-batch
    for k=1:ntrain/trainsize
        updatek = zeros([nh+1,no]);
        updatej = zeros([ni+1,nh]);
        start = (k-1)*trainsize;
        %forward propagate
        netj = train_x(start+1:start+trainsize,:)*w1;
        fnetj = [logistic(netj) ones([trainsize,1])];
        %fnetj = [1.732.*tanh(2/3.*netj) ones([trainsize,1])];
        netk = fnetj*w2;
        fnetk = softmax(netk);
        
        %error calc
        dk = (train_y(:,start+1:start+trainsize)-fnetk');
        u1 = updatek;
        updatek = updatek + (fnetj'*dk');
        dj = (dk'*w2').*fnetj.*(1-fnetj);
        %dj = (dk'*w2').*1.154.*(1-tanh(2/3.*fnetj).*tanh(2/3.*fnetj));
        u2 = updatej;
        updatej = updatej + train_x(start+1:start+trainsize,:)'*dj(:,1:end-1);
        
        w2 = w2 + mom.*u1 + eta.*updatek./trainsize;
        w1 = w1 + mom.*u2 + eta.*updatej./trainsize;
        eta = eta0/(1+i/T);
    end
    time = time+toc;
    % Checking accuracies
    netj = train_x(1:ntrain,:)*w1;
    fnetj = [logistic(netj) ones([ntrain,1])];
    %fnetj = [1.732.*tanh(2/3.*netj) ones([ntrain,1])];
    netk = fnetj*w2;
    fnetk = softmax(netk);
    [~, Idx] = max(fnetk,[],2);
    [~, Idx2] = max(train_y(:,1:ntrain),[],1);
    %for x=1:nval
    %    fprintf('%d %d %d\n',Idx(x),Idx2(x),y_train(ntrain+x));
    %end
    count0 = nnz(Idx==Idx2');
    
    netj = val_x(1:nval,:)*w1;
    fnetj = [logistic(netj) ones([nval,1])];
    %fnetj = [1.732.*tanh(2/3.*netj) ones([nval,1])];
    netk = fnetj*w2;
    fnetk = softmax(netk);
    [~, Idx] = max(fnetk,[],2);
    [~, Idx2] = max(val_y(:,1:nval),[],1);
    %for x=1:nval
    %    fprintf('%d %d %d\n',Idx(x),Idx2(x),y_train(ntrain+x));
    %end
    count1 = nnz(Idx==Idx2');
    
    netj = test_x(1:ntest,:)*w1;
    fnetj = [logistic(netj) ones([ntest,1])];
    %fnetj = [1.732.*tanh(2/3.*netj) ones([ntest,1])];
    netk = fnetj*w2;
    fnetk = softmax(netk);
    [~, Idx] = max(fnetk,[],2);
    [~, Idx2] = max(test_y(:,1:ntest),[],1);
    %fprintf('%d %d %d\n’,Idx,Idx2,y_test);
    count2 = nnz(Idx==Idx2');
    tr_ac(i) = count0/ntrain;
    val_ac(i) = count1/nval;
    te_ac(i) = count2/ntest;
    fprintf('Iteration : %d Train Accuracy : %0.4f Val Accuracy : %0.4f Test Accuracy : %0.4f Time : %0.2f\n',i,count0/ntrain,count1/nval,count2/ntest,time/i);
end

function [g] = logistic(z)
par_res=1+exp(-z);
g=1./par_res;
end

function [g] = tanhh(z)
g = 2/3.*tanh(z);
end

function [softmaxA] = softmax(A)
dim = 2;
s = ones(1, ndims(A));
s(dim) = size(A, dim);
maxA = max(A, [], dim);
expA = exp(A-repmat(maxA, s));
softmaxA = expA ./ repmat(sum(expA,dim), s);
end

function [oh] = onehot(v,c)
oh = zeros(length(v), c);
for i=1:length(v)
    oh(i,v(i)+1) = 1;
end
end