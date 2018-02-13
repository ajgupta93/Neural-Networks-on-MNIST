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

load weights.mat;
eta = 0.001;
dnum = 0.0;
dcalc = 0.0;
count = 0;
epsilon = 0.01;
nlayers = length(w);
n = 100000;
delta = 0.0;
s = 1;
for i=1:nlayers
    for j=1:size(w{i},1)
        for k=1:size(w{i},2)
            %forward propagate
            wp = w;
            wn = w;
            wp{i}(j,k) = wp{i}(j,k)+epsilon;
            wn{i}(j,k) = wn{i}(j,k)-epsilon;

            netj = train_x(s,:)*w{1};
            fnetj = [sigmoid(netj) 1];
            netk = fnetj*w{2};
            y = softmax(netk);

            netj = train_x(s,:)*wp{1};
            fnetj = [sigmoid(netj) 1];
            netk = fnetj*wp{2};
            yp = softmax(netk);

            netj = train_x(s,:)*wn{1};
            fnetj = [sigmoid(netj) 1];
            netk = fnetj*wn{2};
            yn = softmax(netk);

            dnum = calcNumGradient(yp,yn,train_y(:,s),epsilon);

            %error calc
            update = w;
            dk = (train_y(:,s)-y');
            update{2} =(fnetj'*dk');
            dj = (dk'*w{2}').*fnetj.*(1-fnetj);
            update{1} = train_x(s,:)'*dj(:,1:end-1);
            dcalc = -update{i}(j,k);
            delta = delta + abs(dcalc-dnum);
            count = count+1;
        end
    end
end

fprintf('Avg Difference in Gradient : %0.4f     Epsilon square : %0.4f\n',delta/count,epsilon^2);

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

function [numE] = calcNumGradient(yp,yn,t,ep)
yp =log(yp);
yn = log(yn);
Ep = -yp*t;
En = -yn*t;
numE = (Ep-En)/(2*ep);
end

