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
n_iters = 20000;
nh1 = 300;
nh2 = 270;
ni = 784;
no = 10;
eta0 = 0.3;
eta = eta0;
T = 1000000000;
trainsize = 50;
mom = 0.7;

% Initialization considering fan-in
w1 = normrnd(0,1/sqrt(ni),[ni+1,nh1]);
w1(end,:) = normrnd(0,1,[1,nh1]);
w2 = normrnd(0,1/sqrt(nh1),[nh1+1,nh2]);
w2(end,:) = normrnd(0,1,[1,nh2]);
w3 = normrnd(0,1/sqrt(nh2),[nh2+1,no]);
w3(end,:) = normrnd(0,1,[1,no]);

tr_ac = zeros([1,n_iters]);
val_ac = zeros([1,n_iters]);
te_ac = zeros([1,n_iters]);

for i=1:n_iters
    for k=1:ntrain/trainsize
        updatek = zeros([nh2+1,no]);
        updatej = zeros([nh1+1,nh2]);
        updatei = zeros([ni+1,nh1]);
        start = (k-1)*trainsize;
        
        %forward propagate
        neti = train_x(start+1:start+trainsize,:)*w1;
        %fneti = [sigmoid(neti) ones([trainsize,1])];
        fneti = [tanh(1.732.*2/3.*neti) ones([trainsize,1])];
        netj = fneti*w2;
        %fnetj = [sigmoid(netj) ones([trainsize,1])];
        fnetj = [1.732.*tanh(2/3.*netj) ones([trainsize,1])];
        netk = fnetj*w3;
        fnetk = softmax(netk);
        
        %error calc
        dk = (train_y(:,start+1:start+trainsize)-fnetk');
        u1 = mom.*updatek;
        updatek = updatek + (fnetj'*dk');
        %dj = (dk'*w3').*fnetj.*(1-fnetj);
        dj = (dk'*w3').*(1-tanh(fnetj).*tanh(fnetj));
        u2 = mom.*updatej;
        updatej = updatej + fneti'*dj(:,1:end-1);
        %di = (dj(:,1:end-1)*w2').*fneti.*(1-fneti);
        di = (dj(:,1:end-1)*w2').*1.154.*(1-tanh(2/3.*fneti).*tanh(2/3.*fneti));
        u3 = mom.*updatei;
        updatei = updatei + train_x(start+1:start+trainsize,:)'*di(:,1:end-1);
        w3 = w3 + u1+eta.*updatek./trainsize;
        w2 = w2 + u2+eta.*updatej./trainsize;
        w1 = w1 + u3+eta.*updatei./trainsize;
        eta = eta0/(1+i/T);
        
    end
    % Calculate accuracies
    neti = train_x(1:ntrain,:)*w1;
    %fneti = [sigmoid(neti) ones([ntrain,1])];
    fneti = [1.732.*tanh(2/3.*neti) ones([ntrain,1])];
    netj = fneti*w2;
    %fnetj = [sigmoid(netj) ones([ntrain,1])];
    fnetj = [1.732.*tanh(2/3.*netj) ones([ntrain,1])];
    netk = fnetj*w3;
    fnetk = softmax(netk);
    [~, Idx] = max(fnetk,[],2);
    [~, Idx2] = max(train_y(:,1:ntrain),[],1);
    count0 = nnz(Idx==Idx2');
    
    neti = val_x(1:nval,:)*w1;
    %fneti = [sigmoid(neti) ones([nval,1])];
    fneti = [1.732.*tanh(2/3.*neti) ones([nval,1])];
    netj = fneti*w2;
    %fnetj = [sigmoid(netj) ones([nval,1])];
    fnetj = [1.732.*tanh(2/3.*netj) ones([nval,1])];
    netk = fnetj*w3;
    fnetk = softmax(netk);
    [~, Idx] = max(fnetk,[],2);
    [~, Idx2] = max(val_y(:,1:nval),[],1);
    count1 = nnz(Idx==Idx2');
    
    neti = test_x(1:ntest,:)*w1;
    %fneti = [sigmoid(neti) ones([ntest,1])];
    fneti = [1.732.*tanh(2/3.*neti) ones([ntest,1])];
    netj = fneti*w2;
    %fnetj = [sigmoid(netj) ones([ntest,1])];
    fnetj = [1.732.*tanh(2/3.*netj) ones([ntest,1])];
    netk = fnetj*w3;
    fnetk = softmax(netk);
    [~, Idx] = max(fnetk,[],2);
    [~, Idx2] = max(test_y(:,1:ntest),[],1);
    count2 = nnz(Idx==Idx2');
    tr_ac(i) = count0/ntrain;
    val_ac(i) = count1/nval;
    te_ac(i) = count2/ntest;
    fprintf('Iteration : %d     Train Accuracy : %0.4f Val Accuracy : %0.4f    Test Accuracy : %0.4f\n',i,count0/ntrain,count1/nval,count2/ntest);
end

function [g] = sigmoid(z)
par_res=1+exp(-z);
g=1./par_res;
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