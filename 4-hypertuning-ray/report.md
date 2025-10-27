# Hypertuning assignment

## Dataset

For this assignment the CIFAR10 dataset is used. It contains small RGB images of 10 classes, each class has 60000 images (50000 training and 10000 test).

## 1. Type of model

Before fine tuning any model, I want to compare the learning curve of a neural network (NN) with and without residual layers with a convolutional neural network (CNN). Based on the theory, the should learn better with less epochs as it is able the find patterns in the images.

I trained 2 neural networks with 5 layers, one with residual layers added. Then one CNN with 3 layers of 64 filters with kernel size of 3 followed by 2 fully connected layer. In the figures below it is clear that the CNN is able to reach higher accuracy within 5 epochs  -

<table>
    <tr>
        <th>NN</th>
        <th>NN with Residual Layers</th>
        <th>CNN</th>
    </tr>
    <tr>
        <td><img src="images/image-2.png" alt="NN" width="250"/></td>
        <td><img src="images/image-1.png" alt="NN with residual layers" width="250"/></td>
        <td><img src="images/image.png" alt="CNN" width="250"/></td>
    </tr>
</table>

## 2. Grid search for structure of CNN

To get a better feeling for how the CNN should look like, I performed a grid search with the following variables:

```python
{
    "num_fully_connected_layers": tune.grid_search([2,4]),
    "num_conv_layers": tune.grid_search([2,3,4]),
    "filters": tune.grid_search([64,128,152]),
    "kernel_size": tune.grid_search([2,3]),
}
```

As training is not very fast, I only used 3 epochs per model configuration. In general more filters performed better, while more than 2 convolutional layers does not increase performance (see heatmap). Also a kernel size of 3 performs slightly better than 2 and increasing the number of fully connected layers does not improve the validation accuracy.

![alt text](images/image-3.png)

## 3. Hyperband tuning to get optiomal structure of CNN

To explore the possible structures further, I performed a hyperband tuning, with 20 possible samples, setting grace_period to 3 and the max number of epochs to 10, optimizing on validation loss.

```python
{
"hidden_size": tune.randint(254, 512),
"filters": tune.randint(100, 200),
"kernel_size": tune.randint(2, 3),
"padding": tune.randint(0, 1),
}
```

Although the training showed improved performance, the train_loss and validation loss started to diverge around 6 epochs, suggesting the model is overfitting.

Trying to get the model to learn the patterns better, so overfitting can be avoided I added a crop and horizontal flip to the data loader transformer. As visible in the images below, it does not completely remove the overfitting effect, but limits it

<table>
    <tr>
        <td><img src="images/overfitting.png" alt="Overfitting Example 1" width="350"/></td>
        <td><img src="images/overfitting2.png" alt="Overfitting Example 2" width="350"/></td>
    </tr>
    <tr>
        <td align="center">Train vs Validation Loss before applying transformations</td>
        <td align="center">Train vs Validation Loss after applying transformations</td>
    </tr>
</table>

Best configuration before adding transformers:
- hidden_size: 378
- filters: 100
- kernel_size: 2
- padding: 0

Resulting in 72.83% accuracy on the test set in 10 epochs

Best configuration when adding transformers:
- hidden_size: 415
- filters: 146
- kernel_size: 2
- padding: 0

Resulting in 73.18% accuracy on the test set in 10 epochs.

## 4. Improve model hyperparameters with Hyperband scheduler

Trying to improve the performance, I wanted to see if I can implement residual layers, batch normalization and dropout. To try as many possibilities, without having to wait for days, I used the hyperband scheduler. Here I also used different batch_sizes to see if that might influence the accuracy. These are the 5 best performing settings and their performance:

| Trial ID      | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) | Batch Size | Dropout | Filters | Hidden Size | Kernel Size | Conv Layers | FC Layers | Padding | BatchNorm | Residual | Epochs | Learning Rate | Optimizer | Scheduler |  
|--------------|------------|----------|---------------|-------------|------------|---------|---------|-------------|-------------|-------------|-----------|---------|-----------|----------|--------|---------------|-----------|-----------|  
| 075d9_00098  | 0.236      | 0.520    | 91.80         | 84.13       | 64         | 0.20    | 128     | 512         | 3           | 3           | 2         | 1       | True      | True     | 50     | 0.001         | Adam      | ReduceLROnPlateau |  
| 075d9_00028  | 0.218      | 0.644    | 92.47         | 81.26       | 128        | 0.30    | 96      | 256         | 5           | 3           | 1         | 1       | True      | True     | 50     | 0.001         | Adam      | ReduceLROnPlateau |  
| 075d9_00045  | 0.157      | 0.677    | 94.52         | 82.69       | 64         | 0.10    | 128     | 256         | 5           | 3           | 1         | 1       | True      | False    | 50     | 0.001         | Adam      | ReduceLROnPlateau |  
| 075d9_00080  | 0.284      | 0.706    | 89.90         | 79.10       | 128        | 0.20    | 96      | 768         | 2           | 3           | 1         | 1       | False     | False    | 48     | 0.001         | Adam      | ReduceLROnPlateau |  
| 075d9_00016  | 0.398      | 0.715    | 85.94         | 77.47       | 128        | 0.20    | 48      | 512         | 3           | 2           | 2         | 0       | True      | True     | 50     | 0.001         | Adam      | ReduceLROnPlateau |  

To see if the mistakes that are still made by the model is logical, I created a confusion matrix and had a look at some of the mislabeled images:

![Confusion matrix](images/confusion_matrix.png)

![alt text](images/correct_pred.png)

![alt text](images/incorrect_pred.png)

There are some classes that are often mismatched. Like the airplane and ship, that might be due to both having a blue background. And dog, cat and deer, probably due to having similar features like legs, head and body. 

## 5. Comparison with transfer learning

