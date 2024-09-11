# neural-networks: Binary Classification
## 1. Database

**Title**: Parkinson's Disease Data Set; Oxford Parkinson's Disease Detection Dataset  
**Number of Instances**: 196 (including the header with information about column content)  
**Domain**: Life Sciences  
**Number of Attributes**: 23 (including patient names)  
**Missing Values**: None  

The database used contains 195 vocal recordings with 23 different vocal measurements for each, of which 147 patients have Parkinson's disease, and the remaining 48 do not. Each column contains a different vocal measurement, and each row represents a different patient. The "Status" column indicates the presence or absence of the disease, with values of 1 and 0, respectively.

**Format**: CSV  
**Objective**: Differentiating between sick and healthy patients

## 2.Project Structure

### Folders and Files:

- **Activation Functions**
  - **dmsilu.m**: Contains the implementation of the derivative of the MSILU activation function.
  - **msilu.m**: Contains the implementation of the MSILU activation function.
  - **sigmoid.m**: Contains the implementation of the sigmoid activation function.

- **Loos Function**
  - **binaryCrossEntropyLoss.m**: Contains the implementation of the binary cross-entropy loss function, used in this binary classification problem.

- **Methods**
  - **BFGS.m**: Contains the implementation of the BFGS optimization algorithm, used for training the neural network.
  - **GRADIENT.m**: Contains the implementation of the Gradient Descent optimization algorithm, also used for training the model.
- **parkinsons.data**: Contains the dataset used in this project, which includes vocal measurements of patients with and without Parkinson's disease.

  - **template.m**: A script that handles data loading, splitting the data into training and testing sets, applying the methods (BFGS and Gradient Descent), and plotting the results.


## 3. Optimization Algorithms Used

### BFGS Algorithm
BFGS is an iterative optimization algorithm used in convex optimization, particularly for training neural networks. It is classified as a quasi-Newton method and offers a more efficient alternative by approximating the inverse Hessian matrix without calculating and storing the entire Hessian.

In the context of neural network training, the objective function measures the model's performance on training data. The goal is to minimize this loss function by adjusting the neural network's weights. BFGS optimizes the weights iteratively:
- An initial estimate of the inverse Hessian is made, so the first updates are similar to simple gradient descent.
- In each iteration, the algorithm computes the gradient of the objective function, which indicates the direction of the steepest ascent.
- The approximate inverse Hessian is used to determine the direction for the next step to minimize the loss function.
- The weights are updated in this direction, with the step size determined by a constant step (in this case).
- A new gradient is computed, and information from the last two gradients is used to update the inverse Hessian estimate.

BFGS is efficient for binary classification problems and can be a good choice when the dataset is of moderate size, as it converges quickly to a good solution without getting stuck in local minima, unlike simple gradient methods.

### Gradient Descent Method
The Gradient Descent Method is a fundamental optimization algorithm used to minimize objective functions, aiming to find the set of parameters that minimize the loss function, thus indicating the best model for the available data. The process works as follows:
- In each iteration, the algorithm calculates the gradient of the objective function at the current point. The gradient indicates the direction of the steepest ascent.
- The model parameters are updated by taking a step in the opposite direction of the gradient. The step size is determined by the learning rate, a crucial parameter that controls the extent of parameter change in a single update.
- The process of gradient calculation and parameter update is repeated iteratively.
- The algorithm continues iterating until convergence is reached or the maximum number of operations is achieved.

## 4. Results and Comments

During training, BFGS achieved a loss function value of 0.2618, while Gradient Descent reached a slightly higher value of 0.2846. This suggests that BFGS found a weight configuration that fits the training data slightly better.

However, although BFGS performed better on the training data, Gradient Descent had a lower loss value on the test data, 0.2737 compared to 0.3111 for BFGS. This indicates that Gradient Descent generalized better than BFGS when applied to unseen data. The higher test loss value for BFGS might suggest overfitting, where the model learns the training set details too well and doesn't perform as well on new data.

The confusion matrices are identical for both methods, with 5 true positives, 28 true negatives, 2 false positives, and 4 false negatives, showing that both models are equally good at detecting the positive class and also at correctly predicting the negative class.

Additionally, sensitivity and accuracy have the same values for both methods.

### Objective Function vs Iterations and Time:
BFGS (blue) shows oscillations in the decrease of the objective function, indicating more significant adjustments to the weights, while Gradient Descent (red) appears to decrease in a more stable and consistent manner, both in terms of the number of iterations and time. However, BFGS reaches a lower objective function value in fewer iterations.

### Gradient Norm vs Iterations and Time:
BFGS presents large oscillations in the gradient norm, indicating a more variable and possibly exploratory search in the parameter space, while Gradient Descent shows a steady decrease in the gradient norm with no significant oscillations, suggesting stable convergence towards a minimum point.

## 5. Conclusions
In conclusion, Gradient Descent may be considered to have a more predictable and less volatile behavior in terms of weight adjustments, while BFGS seems to take larger steps that can lead to more substantial variations in the objective function and gradient norm.

This may also explain why Gradient Descent performed better on the test data, suggesting that, although BFGS made larger adjustments during training, this did not necessarily lead to better generalization on unseen data. Gradient Descent, with smaller and more stabilized movements, appears to be more robust in avoiding overfitting and achieving good performance on test data.


