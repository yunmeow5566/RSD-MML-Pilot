# Course Unit: Math Foundation

## Learning Goals: 
- Students can build conceptual understanding of the central idea of model training: fitting model parameters to a set of data samples.
- Students understand the loss function as a way to measure how well the model fits the provided data samples.
- Students can understanding the closed‑form approach, recognize the need for gradient descent in real‑world applications
- Students can compare gradient descent to the closed‑form approach through a toy example. 
- Students can understand the process of approximate a function with gradient descent. 
- Students can describe and understand the basic loop of computing loss, gradient and applying adaptive stepsize to update parameters. 
- Students can build matrix form to handle high dimensional data. 
- Students can break down a complicated function into a sequence of simpler functions and explain how linear and activation layers combine to create a more complex function.
- Students can reason about why we need nonlinear functions. 
- Students build mental model about Universal Approximation Theorem: "with enough linear layers and non linear layers, we can model any function within a error range".
- Students can use Pytorch to define matrix of desired values.

## Discussions
- How to solve classification problems with regression models?
- If there is no activiation layer in between, stacking two linear layers is mathematically redundant. Why?
- Choice of loss functions: MSE/MAE/SmoothL1, CrossEntropy, Focal Loss
## Assignments
This is a warmup task for PA2: please try to play with `lessons\unit03-MathFoundation\sample.py`, modify it to get a perfect model.
You can create a new Pull Request to your own homework repository, copy this file there and make changes.

## References
1. Adam optimizer: [paper link](https://arxiv.org/abs/1412.6980)
2. Activation layers: [link](https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/)
3. Logit/logistic regression: [link](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)