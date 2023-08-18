# Machine learning
Machine learning is turning things (data) into numbers, and finding patterns in those numbers.
In machine learning the computer is the one that finds the patterns with some code and math.
## What
Here is an example
To cook a roast chicken there is input which is all the ingridients, then there are rules and instructions to be followed in a certain order, and from that you get the output which is a cooked roast chicken. This is what a human does. 
Machine learing will take in the input and output, and tell you what the rules and instructions are. 
## Why
Machine learning can be used for complex problems. For example learning how to drive can take a lot of time. If we asked any driver to list all the rules that need to be followed, he would not be able to. Whereas a computer can do that, a computer that knows how to drive, can list all the rules perfectly as they have perfect memory.
## When
Machine learning can be used for anything, as long as you can convert it into numbers and program it to find patterns. But that doesn't mean that you should always use it.
### When not to use it
- When you need explainability.
- When the traditional approach is a better option.
- When errors are unacceptable.
- When you don't have much data.
### When to use it
- problems with long list of rules.
- continually changing enviroments.
- discovering insights within large collections of data.
# Machine learning vs Deep learning
Typically machine learning alogrithms are used on structured data, like tables. A popular algorithm is XGBoost.
Deep learning is typically better for unstructured data. A popular way to do this is by using Neural networks.
## Common algorithms for structured data (machine learning)
- Random forest
- Gradient boosted models
- Naive Bayes
- Nearest neighbour
- Support vector machine
- ...many more
## Common algorithms used for unstructured data (deep learning)
- neural networks
- fully connected neural networks
- convolutional neural network
- recurrent neural network
- transformer
- ...many more
# Neural networks
Neural networks are a type of machine learning algorithm. They are inspired by the human brain.
Neural networks are composed of layers
There is an input layer, where you pass in the data as numbers.
There is an output layer, where you get the result as numbers.
There are hidden layers, where the magic happens.
# PyTorch
It's the most popular research deep learning framework (in python)
It can run your code on a single or multiple GPUs
It can access prebuilt models
Originally designed and used in house by meta, it is now open source and it is used by Tesla, Microsoft and OpenAI.
# What is a tensor
A tensor can be almost anything, as a tensor is just numbers. When you have some data, whatever that may be for example images, and you turn that into arrays of numbers, you essentially turned that data into a tensor. When you pass that to a neural network and you get the output which is also a bunch of numbers, then that output is also a tensor.

#### _**READ 01_PYTORCH_FUNDEMENTALS before reading the following notes**_

# Neural Networks in detail
## Structure
Our brains are truly remarkable. They can easily recognize handwritten numbers and letters, even when people write them differently. Imagine if we had to manually program a computer to do this - it would be incredibly complex. That's why we use machine learning. Let's take a closer look at an example: a model that can identify handwritten numbers from 0 to 9.

Think of neural networks as inspired by our brain. A "neuron" in this context is like a container holding a number, called an "activation." In our case, we're using black and white images to represent handwritten numbers. Each neuron corresponds to a grayscale value of a pixel, and this forms the first layer of our neural network. If an image has 784 pixels, then this first layer has 784 neurons, each with an activation value equal to the grayscale of its respective pixel (a number from 0 to 1).

Our goal is to turn an image into a number between 0 and 9. So, the last layer of our neural network contains 10 neurons, each representing a number in that range. These neurons also have activation values between 0 and 1, indicating the system's confidence in its chosen number.

Between these layers, we have hidden layers. In this example, there are 2 hidden layers, each with 16 neurons.

The whole neural network is composed of these layers, starting from the initial 784 pixels (layer 1) and ending with the 10-neuron result (layer 4). Each layer is connected to the next one. For instance, layer 1 connects to layer 2, which then connects to layer 3, and so on. However, in our basic setup, layer 1 isn't directly connected to layer 4. Different neural network structures might allow for that, but we're focusing on a simpler setup.

These connections between layers are assigned weights, which are essentially values. Each connection's weight can be any value, typically within a certain range. Importantly, weights can also be negative.

## Calculating the value of each neuron
To compute the activation value of a connected neuron in the next layer, we multiply the activation values from the first layer by the weight linked to that connection. For example, each neuron in the first layer connects to every neuron in the second layer. This means a second-layer neuron has connections from all 784 first-layer neurons. To calculate its value, we multiply each first-layer neuron's value by its corresponding weight, sum everything up, and then adjust the result to fit between 0 and 1. This is often done using the sigmoid function. Additionally, each neuron can have a bias, which is a value added to the sum before applying the sigmoid function.

In maths if each weight is $w_k._n$ where $w$ is the weight, $n$ is the index of the neuron in the first layer and $k$ is the index of the neuron that it is going into or in other words the neuron in the next layer (the second layer) and each activation value for the first layer of neurons is represented by $a_n$. The mathematical function for each neuron in the second layer would look something like $w_k._0* a_0 + w_k._1 * a_1 ... + w_k._n * a_n + b_k$ wrapped in the sigmoid function.

A simpler way to compute the next layer involves arranging the activation values of the first layer into a tensor of shape [1, 784], let's call this tensor $A$. Similarly, the connections into the next neuron are organized into a tensor of shape [784, 16] (as each one of the 784 neurons has 16 connections) let's call this tensor $B$. This setup allows us to multiply these tensors with a resulting tensor of shape [1, 16], but we also need to add biases, weo do this by arranging the biases into a tensor of shape [1, 16] let's call this tensor $C$, and apply the sigmoid function to the result. When we put this together we get $AB + C$ or in other terms $mx + c$.

While we often think of neurons as representing numbers, it's perhaps more accurate to view them as mathematical functions. Given input values from the previous layer and connection weights, each neuron calculates a value to determine its own activation value.

## How good is our model
Carring on from before, we now have a function for each neuron in each layer.
To get started we will assign random values to each weight and bais. We now have a functional neural network, we can pass in an image, and it should be able to give us a result, however, that result will be completly random and wrong, so how can we train it?


So what we do is calculate the cost, using the cost function, to do so, we need to know what the result should have been. For each neuron in the final layer, we have a result, we have a value, we also have the true result, which let's say is 3, to convert that into appropriate data, that would look like a tensor with shape [1, 9] and all values are 0 except for the 3rd value which is 1. We can then use the cost function to calculate the cost, which is the difference between the result and the true result => squared so $(r_1 - r_2)^2$ where are $r1$ is the resut and $r2$ is the true result. We then add everything up. The smaller the value the closer you are to the actual value (it cannot be negative). So now we know how to bad the current version of the computer is, but we need to improve it. 

### How to improve a model
#### Gradient Descent
Currently we have 784 inputs and 10 outputs, we then have a cost, which is how bad/good our model currently is. 
Let's temporarly simplfy this to 1 input and 1 output, we can then plot the cost against the input, and we get a graph that looks like a parabola. We can then find the minimum of this graph, which is the lowest point, and that is the best value for the input.
When we begin we will be at a random point on that graph, to go down we would calculate the gradient, and move towards the lowest point, step by step.
However the graph is not always a parabola and it can be more complex for example it can be a saddle or a plynomial function, and we want the global minimum, which is the lowest point on the graph. We cannot always get that, wether or not you will get it depends on where on that graph you begin. Especially when dealing with more complex situation for example, simply adding another input, so two inputs, will make the graph 3D, and it will be a surface, and it will be harder to find the global minimum.
Calculating the gradient and going down the slope step by step is what we call gradient descent.

#### Back propagation
This is about how to actually calculate the gradient we talked about earlier.
let's look at an example where you pass in an image of a 2, and the result is of course all over the place and random.
What we now want is for all neuron values that are not 2 to be lowered, and for the number 2 neuron to go up if it isn't already a 1.
So if we zoom into neuron 2, that value is defined form the weights and it's bias. So we can do three things here, increase the bias, increase the weight, or change the activation value of the previous layer, by chaing it's biases and weights.
However we dont just want to increase or decrease everything, we want to increase the best weight, the one that has the highest effect, and that depends on which neuron from the previous layer has the highest value.
It is important to also remmeber that weights can be negative, so to help increase the value of the number 2 neuron, we would need to decrease the activation value of everything with a negative weight, and increase the activation value of everything with a positive weight. However we dont have control over the activation value, but rather it's weights and biases.
Another important thing to keep in mind, is that all these changes will help you recognize a two, but we want to recognize all the numbers.
So we will have a loot at all the other numbers and see what they want to get done on this neural network. Since the image given was a two, all other neuron will want to decrease their activation value.
It is important to note that these nudges have a size and propotion that depends on the activation value of the node.
You add all of these nudges together and find the average, you now have a single value for each node and you want to apply the same process again but for the layer before.
To be accurate you would need to do this for all training data (each image in our case) find the average and then apply the step, but that would take a lot of power, what you usually do is take a step for each data (image), which is much faster, and it will still have a similar result. Although you would be going down the slope (from the graph we talked about earlier) as a drunk man, in other words taking the wrong path, slightly longer (in terms of distance but not speed) and get to the same destination but faster, rather than precisely going down the slope.