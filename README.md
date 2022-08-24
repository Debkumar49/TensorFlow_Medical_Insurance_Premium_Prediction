# Neural Network Regression with TensorFlow - Medical Insurance Premium Prediction
Medical Insurance Predictions
 
 ## What is Neural Network ?
 It’s a technique for building a computer program that learns from data. It is based very loosely on how we think the human brain works. First, a collection of software “neurons” are created and connected together, allowing them to send messages to each other. Next, the network is asked to solve a problem, which it attempts to do over and over, each time strengthening the connections that lead to success and diminishing those that lead to failure.
 ## Linear Regression
Linear Regression is a supervised learning technique that involves learning the relationship between the features and the target. The target values are continuous, which means that the values can take any values between an interval. For example, 1.2, 2.4, and 5.6 are considered to be continuous values. Use-cases of regression include stock market price prediction, house price prediction, sales prediction, and etc.
 
 ## What are the Steps followed in the Project?
 #### - Input shapes and output shapes
          - `X`: features/data (inputs)
          - `y`: labels (outputs)
        - Creating custom data to view and fit
        - Steps in modelling
          - Creating a model
          - Compiling a model
            - Defining a loss function
            - Setting up an optimizer
            - Creating evaluation metrics
          - Fitting a model (getting it to find patterns in our data)
        - Evaluating a model
          - Visualizng the model (\"visualize, visualize, visualize\")
          - Looking at training curves
        - Saving a model (so we can use it later)
        
## Architecture of Regression Neural Network.

| **Hyperparameter**  | **Typical Values** |
| ------------- | ------------- |
| Input layer shape  | Same shape as number of features (e.g. 3 for # bedrooms, # bathrooms, # car spaces in housing price prediction)  |
| Hidden layer(s)  | Problem specific, minimum = 1, maximum = unlimited  |
| Neurons per hidden layer  | Problem specific, generally 10 to 100  |
| Output layer shape  | Same shape as desired prediction shape (e.g. 1 for house price)  |
| Hidden activation  | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit)  |
| Output activation  | None, ReLU, logistic/tanh  |
| Loss function  | [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (mean square error) or [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) (mean absolute error)/Huber (combination of MAE/MSE) if outliers  |
| Optimizer  | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)  |

## Split data into training/test set
* `Training set` - the model learns from this data, which is typically 70-80% of the total data available (like the course materials you study during the semester).

* `Validation set` - the model gets tuned on this data, which is typically 10-15% of the total data available (like the practice exam you take before the final exam).

* `Test set` - the model gets evaluated on this data to test what it has learned, it's typically 10-15% of the total data available (like the final exam you take at the end of the semester).



## Steps in Modelling with TensorFlow
In TensorFlow, there are typically 3 fundamental steps to creating and training a model.

1.) `Creating a model` - piece together the layers of a neural network yourself (using the Functional or Sequential API) or import a previously built model (known as transfer learning).

2.) `Compiling a model` - defining how a models performance should be measured (loss/metrics) as well as defining how it should improve (optimizer).

3.) `Fitting a model` - letting the model try to find patterns in the data (how does X get to y).

## Evaluating Predictions
Two of the main metrics used for regression problems are:

`Mean absolute error (MAE)` - the mean difference between each of the predictions.

`Mean squared error (MSE)` - the squared mean difference between of the predictions (use if larger errors are more detrimental than smaller errors).


To see the code and for pull request(to improve code) click on the .ipynb file on the repo   **or**  [**click here**](https://github.com/Debkumar49/TensorFlow_Medical_Insurance_Premium_Prediction/blob/main/Medical_Insurance_Premium_Prediction.ipynb) to see the code 

