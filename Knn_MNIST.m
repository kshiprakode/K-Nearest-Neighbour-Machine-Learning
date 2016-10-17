% Machine Learning
% Kshipra Kode

clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

% % Vector that keeps a track of the accuracy returned
accuracy_vector = zeros(10,1);
data_points = logspace(2,4,10);
i = 1;

% Function that finds the accuracy when k = 1 and the train dataset is
% between 30 to 10000

for v = data_points
    fprintf('\nComputing Accuracy for k = 1 and train data set = %d',round(v));
    [acc,acc_av] = kNN(images_train(1:round(v),:), labels_train, images_test(1:1000,:), labels_test, 1);
    accuracy_vector(i) = acc_av;
    fprintf('\nAccuracy for train data set %d is %f\n\n',round(v),accuracy_vector(i));
    i = i+1;
end

% Plotting the accuracy of the values received
disp('Creating plot for k = 1 and train_data range from 30 to 10000')
figure();
plot(data_points,accuracy_vector);
axis([100 10000 0.5 1]);
title('Accuracy when k = 1 and random data points between 30 to 10000');
ylabel('Accuracy');
xlabel('Size of train data set');


% Function that finds the accuracy when k = [1 2 3 5 10] and the train dataset is
% between 30 to 10000

k = [1 2 3 5 10];
figure();
for j = k
    i = 1;
    for v = data_points
        fprintf('\nComputing Accuracy for k = %d and train data set = %d',j,round(v));
        [acc,acc_av] = kNN(images_train(1:round(v),:), labels_train, images_test(1:1000,:), labels_test, j);
        accuracy_vector(i) = acc_av;
        fprintf('\nAccuracy for train data set %d is %f\n\n',round(v),accuracy_vector(i));     
        i = i+1;
    end

    % Plotting the accuracy of the values received
    plot(data_points,accuracy_vector);
    axis([100 10000 0.5 1]);
    hold all;
end
disp('Creating plot for k = [1 2 3 5 10] and train_data range from 30 to 10000')
title('Accuracy when K in [1 2 3 5 10] with data points between 30 to 10000');
ylabel('Accuracy');
xlabel('Size of train data set');
legend('K = 1','K = 2','K = 3','K = 5','K = 10');


% Function that finds the accuracy when k = [1 2 3 5 10] and the train dataset is
% between is 1000 d

accuracy_vector = zeros(5,1);
k = [1 2 3 5 10];
i = 1;
for j = k
    fprintf('\nComputing Accuracy for k = %d and train data set = 1000 and test data set = 1000',j);
    [acc,acc_av] = kNN(images_train(1:1000,:), labels_train, images_train(1001:2000,:), labels_train(1001:2000,:), j);
    accuracy_vector(i) = acc_av;
    fprintf('\nAccuracy for train data set 1000 and test data set is 1000 is %f\n\n',accuracy_vector(i));     
    i = i+1;
end

disp('Creating plot for k = [1 2 3 5 10] and train data range 1000 and test data 1000')
% Plotting the accuracy of the values received
figure();
plot(k,accuracy_vector);
title('Accuracy for K in [1 2 3 5 10] with train data set of 1000 and validation data set of 1000');
ylabel('Accuracy');
xlabel('K');
[ac,index] = sort(accuracy_vector);
fprintf('\nMaximum Accuracy for  k = %d\n',round(k(index(5))));

% Function that finds the kNN of given test data
function [acc, acc_av] = kNN(images_train, labels_train, images_test, labels_test, k)
    
    test_size = size(images_test,1);
    train_size = size(images_train,1);
    
    %Counters to keep a track of the predictions made by the model
    classification_counter = zeros(10,1);
    accurate_predictions = 0;
    prediction_counter = zeros(10,1);
    
    % For loop that takes each point in test data, finds the distance from the train data points 
    for i = 1:test_size
        
        distances = zeros(train_size,1);
        
        % Loop that finds distance between the test data and all the train
        % data points
        for n = 1:train_size
            distances(n) = distance(images_test(i,:),images_train(n,:));
        end
        
        % Sorting the distances to find the closet one
        [~, index] = sort(distances);
        
        % Find the labels of the train data that have closest distances
        kNN_labels = zeros(k, 1);
        for j = 1:k
            kNN_labels(j) = labels_train(index(j));
        end
        
        % Find the classifier highly predicted
        prediction = mode(kNN_labels);
        
        % Update the counters to calculate accuracy
        if(prediction == labels_test(i))
            accurate_predictions = accurate_predictions + 1;
            prediction_counter(prediction + 1) = prediction_counter(prediction + 1) + 1;
        end
        
        classification_counter(labels_test(i) + 1) = classification_counter(labels_test(i) + 1) + 1;
    
    end
        
        % Return the values for the average accuracy and the accuracy for
        % each of the digits
        acc_av = accurate_predictions / test_size;
        acc = prediction_counter./classification_counter;
        
end

% Function to compute the Euclidean distance between two vector points
function dist = distance(point_1, point_2)
        dist = sqrt(sum((point_1 - point_2).^2));
end