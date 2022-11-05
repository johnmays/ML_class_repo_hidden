# Written Homework 4

Names and github IDs (if your github ID is not your name or Case ID):

James McHargue, John Mays, Zaichuan You

18.	Show that the set $C=$ \{ $x|Ax\geq b$ \}, $A \in R^{m\times n}$, $x \in R^n$, $b \in R^m$, is a convex set. Note that this describes the constraint set of a linear program. (10 points)

Answer: 

_claimed by John_

19.	A function $f$ is said to have a global minimum at $x$ if for all $y$, $f(y) \geq f(x)$. It is said to have a local minimum at $x$ if there exists a neighborhood $H$ around $x$ so that for all $y$ in $H$, $f(y)\geq f(x)$. Show that, if $f$ is convex, every local minimum is a global minimum. [Hint: Prove by contradiction using Jensen’s inequality.] (10 points)

Answer: 

20.	Consider the LP: $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$, where $T$ is the transpose, $A$ is the 4x2 matrix: \[ 0 −1; −1 −1; −1 2; 1 −1\], $b$ is a 4x1 vector \[−5; −9;0; −3\] and $c$ is a 2x1 vector \[−1; −2\]. (a) Draw the feasible region in $R^2$. (b) Draw the contours of $c^Tx =−12$, $c^Tx =−14$ and $c^Tx =−16$ and determine the solution graphically. (10 points)

Answer: 

21.	Consider the primal linear program (LP): $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$ and its dual: $\max b^Tu$ s.t. $A^Tu \leq c, u \geq 0$. Prove that for any feasible $(x,u)$ (i.e. $x$ and $u$ satisfying the constraints of the two LPs), $b^Tu \leq c^Tx$. (10 points)

Answer: Looking at the laws of matrix manipulation, this problem actually becomes fairly simple. The key is combining the two sets of constraints to essentially rewrite $b^Tu$ as $c^Tx$:

- $b^Tu = u^Tb$
- Since $b \leq (Ax)$, $b^Tu \leq u^T(Ax)$.
    - $u^T(Ax) = (u^TA)x$ by the associative property of matrix multiplication
    - $= (A^Tu)^Tx$ by a transpose rule of matrices
    - $\leq (c^T)x$ by the constraint of the dual problem
- So, $b^Tu \leq u^T(Ax) \leq (c^T)x$
- By the transitive property, $b^Tu \leq (c^T)x$


22.	Derive the backpropagation weight updates for hidden-to-output and input-to-hidden weights when the loss function is cross entropy with a weight decay term. Cross entropy is defined as $L(\mathbf{w})=\sum_i y_i\log{(\hat{y}_i)}+(1-y_i)\log{(1-\hat{y}_i)}$ , where $i$ ranges over examples, $y_i$ is true label (assumed 0/1) and $\hat{y}_i$  is the estimated label for the $i^{th}$ example. (10 points)

Answer: The hidden-to-output weights are straightforward to calculate. Since the overall loss is a summation across all output nodes, and each output only appears in one part of the sum, to find the derivative of the loss with respect to one output node, we only have to consider its specific part of the sum:

$L(i) = y_ilog(\hat{y_i}) + (1 - y_i)log(1 - \hat{y_i})$

The derivative of the loss with respect to a given node output $x^L_i$ is therefore

$\frac{dL}{dx^L_i} = \frac{y_i}{x^L_i} + \frac{y_i-1}{1-x^L_i}$

In a similar way to how we derived backpropagation updates in class, we can derive the output weight gradient $\frac{dL}{dw^L_{ij}}$ using the chain rule:

$\frac{dL}{dw^L_{ji}} = \frac{dL}{dx^L_j} \frac{dx^L_j}{dn^L_j} \frac{dn^L_j}{dw^L_{ji}}$

We know the first term from above; the second term is the derivative of the sigmoid function, $h’(x) = h(x)(1 - h(x)$, and the third term is the ‘spiking rate’ of the neuron to which the weight $w^L_{ji}$ is applied. Additionally, because we’re including a weight decay term, the loss function should have some direct derivative with respect to $w^L_{ji}$. If we assume that the weight decay term is $\frac{1}{2}|w|^2$, then this derivative is actually equal to $w^L_{ji}$.

$\frac{dL}{dw^L_{ji}} = (\frac{y_i}{x^L_i} + \frac{y_i-1}{1-x^L_i}) (h(n^L_j)(1-h(n^L_j))) (x_i^{L-1}) + w^L_{ji}$

Since the input-to-hidden weight gradients are dependent only on the gradients of later weights (rather than the explicit loss function), we can express them in essentially the same way as in class for squared loss. We still derive them using the chain rule, except for that $\frac{dL}{dx^l_j}$ is calculated as a sum across the gradients of all downstream nodes:

$\frac{dL}{dw^l_{ji}} = (\sum_{k} \frac{dL}{dw_{kj}^{l+1}} \frac{w_{kj}^{l+1}}{x_j^l}) (h(n^l_j)(1-h(n^l_j))) (x_i^{l-1}) + w^L_{ji}$


23.	Consider a neural network with a single hidden layer with sigmoid activation functions and a single output unit also with a sigmoid activation, and fixed weights. Show that there exists an equivalent network, which computes exactly the same function, where the hidden unit activations are the $\tanh$ function described in class, and the output unit still has a sigmoid activation. (10 points)

Answer: The key to this problem is demonstrating that the tanh() function is a linearly-scaled version of the sigmoid function. This is somewhat hard to see at first, but becomes obvious once you derive it:

$tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$

$tanh(x) = \frac{(e^{2x} + 1) - (2)}{e^{2x} + 1}$

$tanh(x) = \frac{e^{2x} + 1}{e^{2x} + 1} - \frac{2}{e^{2x} + 1}$

$tanh(x) = 1 - \frac{2}{e^{2x} + 1}$

$tanh(x) = 1 - 2\sigma(-2x)$

$tanh(x) = 1 - 2(1 - \sigma(2x))$

$tanh(x) = 2\sigma(2x) - 1$

So, the takeaway of this is:

$tanh(0.5x) = 2\sigma(x) - 1$

Call the original vector of outputs from the hidden layer $z$. Also call the matrix of input-hidden weights $W_{hi}$, the vector of hidden layer biases $b_h$, the vector of hidden-output weights $w_{oh}$, and the output bias $b_o$. We define the structure of the original network as:

$z = \sigma(W_{hi} x + b_h)$

$y = \sigma(w_{oh} \cdot z + b_o)$

Based on the derivation we found above, we know that

$tanh(0.5(W_{hi} x + b_h)) = 2\sigma(W_{hi} x + b_h) - 1$

$tanh(0.5(W_{hi} x + b_h)) = 2z - 1$

If we were to introduce a new weight matrix $W’_hi$ and bias vector $b’_h$ that are equal to the originals, but scaled by 0.5, then we could say that

$tanh(W’\_{hi} x + b’_h) = 2z - 1$

So, with these new weights, biases, and the tanh() function, **the resulting hidden vector is equal to twice the original, with all values reduced by 1.** By adjusting the output weights and the output bias term, we can reconstruct the same output $y$ as the original using this vector. 

In the original network, we get the output through a dot product of the hidden vector and the output weights.

$y = \sigma(\sum\_i (w\_{oh, i}z_i) + b_o)$

Let $w’\_{oh} = 0.5w_{oh}$, and let $b’\_o = b_o + \sum_i 0.5w_{oh, i}$. All values in our 'transformed' hidden vector are equal to those in the original, doubled, minus 1. If we use these new parameters with the adjusted hidden vector,

$\sigma(\sum\_i (w’\_{oh, i}(2z_i - 1)) + b’\_o)$

$= \sigma(\sum\_i (0.5w\_{oh, i}(2z_i - 1)) + b_o + \sum_i 0.5w_{oh, i})$

$= \sigma(\sum\_i (w\_{oh, i}z_i - 0.5w\_{oh, i}) + b_o + \sum_i 0.5w_{oh, i})$

$= \sigma(\sum\_i (w\_{oh, i}z_i) - \sum_i(0.5w\_{oh, i}) + b_o + \sum_i 0.5w_{oh, i})$

$= \sigma(\sum\_i (w\_{oh, i}z_i) + b_o)$

$= y$

We recover the original output.

In summary, setting $W’\_{hi} = 0.5W\_{hi}$, $b’\_h = 0.5b_h$, $w’\_{oh} = 0.5w_{oh}$, and $b’\_o = b_o + \sum_i 0.5w_{oh, i}$ with the tanh() function computes the same output.


24.	Draw an artificial neural network structure which can perfectly classify the examples shown in the table below. Treat attributes as continuous. Show all of the weights on the edges. For this problem, assume that the activation functions are sign functions instead of sigmoids. Propagate each example through your network and show that the classification is indeed correct.
(10 points)
 
|x1	|x2	|Class|
|---|---|-----|
|−4	|−4	|−|
|−1	|−1	|+|
| 1	| 1	|+|
| 4|  4	|−|

Answer:

25.	Using R/Matlab/Mathematica/python/your favorite software, plot the decision boundary for an ANN with two inputs, two hidden units and one output. All activation functions are sigmoids. Each layer is fully connected to the next. Assume the inputs range between −5 to 5 and fix all activation thresholds to 0. Plot the decision boundaries for  the weights except the thresholds randomly chosen between (i) (−10,10), (ii) (−3,3), (iii) (−0.1,0.1) (one random set for each case is enough). Use your plots to show that weight decay can be used to control overfitting for ANNs. (If you use Matlab, the following commands might be useful: meshgrid and surf). (20 points)

Answer:
_claimed by John_

26.	When learning the weights for the perceptron, we dropped the *sign* activation function to make the objective smooth. Show that the same strategy does not work for an arbitrary ANN. (Hint: consider the shape of the decision boundary if we did this.)  (10 points)

Answer:
_claimed by John_
