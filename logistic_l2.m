images_train = loadMNISTImages('train-images-idx3-ubyte');
labels_train = loadMNISTLabels('train-labels-idx1-ubyte');

images_test = loadMNISTImages('t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

d1 = 2;
d2 = 3;

tr_im = images_train(:,labels_train==d1 | labels_train == d2)';
tr_l = labels_train(labels_train==d1 | labels_train == d2);

te_im = images_test(:,labels_test==d1 | labels_test == d2)';
te_l = labels_test(labels_test==d1 | labels_test == d2);

tr_l(tr_l==d1) = 1;
tr_l(tr_l==d2) = 0;

te_l(te_l==d1) = 1;
te_l(te_l==d2) = 0;

ntrain = 4000;
nval = 400;
ntest = 2000;

train_x = [ones(ntrain,1),tr_im(1:ntrain,:)]';
train_y = tr_l(1:ntrain)';

val_x = [ones(nval,1),tr_im(ntrain+1:ntrain+nval,:)]';
val_y = tr_l(ntrain+1:ntrain+nval)';

test_x = [ones(ntest,1),te_im(1:ntest,:)]';
test_y = te_l(1:ntest)';


ww = zeros([5,n_iters]);


n_iters = 150;
eta0 = 0.001;
T = 2;
lambda = 100;
% figure,
% hold on
for j=1:5
    iters = 0;
    grad = zeros([1,785]);
    w = zeros([1,785]);
    tr_ac = zeros([1,n_iters]);
    val_ac = zeros([1,n_iters]);
    te_ac = zeros([1,n_iters]);

    e1 = zeros([1,n_iters]);
    e2 = zeros([1,n_iters]);
    e3 = zeros([1,n_iters]);

    costJtr = 0;
    costJva = 0;
    costJte = 0;

    while(iters<n_iters)
        p = sigmoid(w*train_x);
        p_val = sigmoid(w*val_x);
        p_test = sigmoid(w*test_x);
        
        grad = 1/ntrain*(( p - train_y ) * train_x');
        grad(2:end) = grad(2:end)+(lambda/ntrain).*w(2:end);
        w = w - grad;
        iters = iters+1;
        costJtr = 0;
        costJva = 0;
        costJte = 0;
        for i=1:ntrain
            costJtr = costJtr - log(p(i)^train_y(i) * (1 - p(i))^(1 - train_y(i)));
        end
        for i=1:nval
            costJva = costJva - log(p_val(i)^val_y(i) * (1 - p_val(i))^(1 - val_y(i)));
        end
        for i=1:ntest
            costJte = costJte - log(p_test(i)^test_y(i) * (1 - p_test(i))^(1 - test_y(i)));
        end
        
        e1(iters) = costJtr/ntrain+(sum(w(2:end).^2)*lambda)/ntrain;
        e2(iters) = costJva/nval+(sum(w(2:end).^2)*lambda)/nval;
        e3(iters) = costJte/ntest+(sum(w(2:end).^2)*lambda)/ntest;
        
        p = round(p);
        p_val = round(p_val);
        p_test = round(p_test);
        tr_ac(iters) = 1-nnz(p==train_y)/ntrain;
        val_ac(iters) = 1-nnz(p_val==val_y)/nval;
        te_ac(iters) = 1-nnz(p_test==test_y)/ntest;
        
        ww(j,iters) = norm(w,2);
        
        if(iters>3 && e2(iters)>e2(iters-1) && e2(iters-1)>e2(iters-2) && e2(iters-2)>e2(iters-3))
            break;
        end
    end
%    plot(1:iters,1-tr_ac(1:iters));
    w = w(2:end);
    w = reshape(w,[28,28]);
    w = imresize(w,[100, 100]);
    figure,imagesc(w),colormap(jet),colorbar;
    lambda = lambda/10;
end
% legend('100','10','1','0.1','0.01');
% xlabel('No of Iterations');
% ylabel('% Correct');
% hold off;

figure,
hold on,
plot(1:5,ww(1,1:5));
plot(1:100,ww(2,1:100));
plot(1:80,ww(3,1:80));
plot(1:80,ww(4,1:80));
plot(1:80,ww(5,1:80));
legend('100','10','1','0.1','0.01');
xlabel('No of Iterations');
ylabel('Weight Vector Length');
hold off;

range = 1:iters;

figure,
hold on,
plot(range,tr_ac(range),'r-');
plot(range,val_ac(range),'g-');
plot(range,te_ac(range),'b-');
legend('Train','Hold-Out','Test');
xlabel('Iterations');
ylabel('Error rate');
hold off;

figure,
hold on,
plot(range,e1(range),'r-');
plot(range,e2(range),'g-');
plot(range,e3(range),'b-');
legend('Train','Hold-Out','Test');
xlabel('Iterations');
ylabel('Loss');
hold off;


%close all;
function [g] = sigmoid(z)
par_res=1+exp(-z);
g=1./par_res;
end
