# Written Homework 4

Names and github IDs (if your github ID is not your name or Case ID):

James McHargue, John Mays, Zaichuan You

18.	Show that the set $C=$ \{ $x|Ax\geq b$ \}, $A \in R^{m\times n}$, $x \in R^n$, $b \in R^m$, is a convex set. Note that this describes the constraint set of a linear program. (10 points)

Answer: 

To begin, I will define a convex set: A set $\mathbf{C}$ is convex if, for any two members, $x_1$ and $x_2$, $x_3 = \lambda x_1 + (1-\lambda) x_2$ is also in $\mathbf{C}$, given $0 \geq \lambda \geq 1$.

So, let's take two arbitrary vectors $x_1$ and $x_2$ belonging to $\mathbf{C} = \textbraceleft x|Ax \geq b \textbraceright$.

We define a third member (that would be in the set if the set is convex) as $x_3 = \lambda x_1 + (1-\lambda) x_2$.  Let's assume it is not in the set, which would imply that $Ax_3 < b$ and we'll start there.

Let's also take advantage of our definitions to say that, since $Ax_1$ and $Ax_2$ are $\geq b$, $Ax_1 = ( b + c)$ where all entries of $c$ must be $\geq 0$, and $Ax_2 = ( b + d)$ where all entries of $d$ must be $\geq 0$.

So...

$Ax_3 < b \implies$

$A(\lambda x_1 + (1-\lambda) x_2) < b \implies$

$\lambda A x_1 + A x_2 -\lambda A x_2 < b \implies$

$\lambda (b+c) + b+d -\lambda (b+d) < b \implies$

$\lambda (b+c) + d -\lambda (b+d) < 0 \implies$

$\lambda b + \lambda c + d -\lambda b -\lambda d  < 0 \implies$

$\lambda c + d -\lambda d  < 0 \implies$

$\lambda c + (1-\lambda) d < 0$

We also know that $\lambda$ is between $0$ and $1$, which means that $(1-\lambda)$ must also be between $0$ and $1$.

Therefore, we have the equation

$(\text{nonnegative coefficient})(\text{nonnegative-entry vector})+(\text{nonnegative coefficient})(\text{nonnegative-entry vector}) < 0$

which is an impossible contradiction.

The conclusion that $x_3 = \lambda x_1 + (\lambda - 1) x_2$ does not belong to $\mathbf{C}$ leads to a contradiction.  Therefore, $x_3$ must belong to $\mathbf{C}$.  

We proved that for any two $x$'s, $x_1$ and $x_2$, $x_3 = \lambda x_1 + (\lambda - 1) x_2$ must belong to the set $\mathbf{C} = \textbraceleft x|Ax \geq b \textbraceright$.

**Therefore, the set $\textbraceleft x|Ax \geq b \textbraceright$ must be convex.**

19.	A function $f$ is said to have a global minimum at $x$ if for all $y$, $f(y) \geq f(x)$. It is said to have a local minimum at $x$ if there exists a neighborhood $H$ around $x$ so that for all $y$ in $H$, $f(y)\geq f(x)$. Show that, if $f$ is convex, every local minimum is a global minimum. [Hint: Prove by contradiction using Jensen’s inequality.] (10 points)

Answer: 
Given that we have a subset which $H \subseteq X$ where in that set f(x) is a local minimum. Assume outside this set H we have a $f(x_g) < f(x)$. By the definition of convexity we have:

$f(\lambda x_g + (1-\lambda)x) \leq \lambda f(x_g)+(1-\lambda)f(x)$

Since $f(x_g) < f(x)$ we have:

$\lambda f(x_g)+(1-\lambda)f(x) \leq \lambda f(x)+(1-\lambda)f(x)=f(x)$

Combine them we have:

$f(\lambda x_g + (1-\lambda)x) \leq \lambda f(x_g)+(1-\lambda)f(x) \leq \lambda f(x)+(1-\lambda)f(x)=f(x)$

In this case if we set $\lambda$ close enough to 0 until we observe that $\lambda x_g + (1-\lambda)x \in H$, the equation give us a result of $f(\lambda x_g + (1-\lambda)x) < f(x)$. However by our definition to f(x), $f(x) \leq f(\lambda x_g + (1-\lambda)x)$.
This is a contradiction and by proving this contradiction we proved that there is no such $f(x_g)$ exist. Which at the same time proved that f(x) is a global minmum.

20.	Consider the LP: $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$, where $T$ is the transpose, $A$ is the 4x2 matrix: \[ 0 −1; −1 −1; −1 2; 1 −1\], $b$ is a 4x1 vector \[−5; −9;0; −3\] and $c$ is a 2x1 vector \[−1; −2\]. (a) Draw the feasible region in $R^2$. (b) Draw the contours of $c^Tx =−12$, $c^Tx =−14$ and $c^Tx =−16$ and determine the solution graphically. (10 points)

Answer: 

The blue part denote the feasible region and red part denote the non-feasible region.
![Figure_1](https://user-images.githubusercontent.com/89466889/201288439-c3036a77-313f-4c9f-bde6-39a60c685034.png)

![Figure_2](https://user-images.githubusercontent.com/89466889/201288505-a16ea69f-f1a5-42a3-b73e-1c4f63c7080a.png)

In this case if we want to minimize $c^Tx$ we will pick [4,5] because it is the only points lies on the vertex $c^Tx =−14$. The problem is convex so there will not be any other solution which has smaller $c^Tx$ than point [4,5].


21.	Consider the primal linear program (LP): $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$ and its dual: $\max b^Tu$ s.t. $A^Tu \leq c, u \geq 0$. Prove that for any feasible $(x,u)$ (i.e. $x$ and $u$ satisfying the constraints of the two LPs), $b^Tu \leq c^Tx$. (10 points)

Answer: Looking at the laws of matrix manipulation, this problem actually becomes fairly simple. The key is combining the two sets of constraints to essentially rewrite $b^Tu$ as $c^Tx$:

- $b^Tu = u^Tb$
- Since $b \leq (Ax)$, $b^Tu \leq u^T(Ax)$.
    - $u^T(Ax) = (u^TA)x$ by the associative property of matrix multiplication
    - $= (A^Tu)^Tx$ by a transpose rule of matrices
    - $\leq (c)^Tx$ by the constraint of the dual problem
- So, $b^Tu \leq u^T(Ax) \leq (c)^Tx$
- By the transitive property, $b^Tu \leq (c)^Tx$


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

$\frac{dL}{dw^l_{ji}} = (\sum_{k} \frac{dL}{dw_{kj}^{l+1}} \frac{w_{kj}^{l+1}}{x_j^l}) (h(n^l_j)(1-h(n^l_j))) (x_i^{l-1}) + w^l_{ji}$


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

![WeChat Image_20221111160645](https://user-images.githubusercontent.com/89466889/201430860-99deb587-ebd2-43d1-8418-811175c80026.png)


Here is a demo of network haveing given example as inputs

|x1|x2|h1|h2|h3|h4|y|
|--|--|---|---|---|---|---|
|-4|-4|-4*1<2=>-1|-4*1<-2=>-1|1+1>-2=>1|-1-1=-2=>-1|1-1<1=>-1|
|-1|-1|-1*1<2=>-1|-1*1>-2=>1|1-1>-2=>1|-1+1>-2=>1|1+1>1=>1|
|1|1|1*1<2=>-1|1*1>-2=>1|1-1>-2=>1|-1+1>-2=>1|1+1>1=>1|
|4|4|4*1>2=>1|4*1>-2=>1|-1-1=-2=>-1|1+1>-2=>1|-1+1<1=>-1|

25.	Using R/Matlab/Mathematica/python/your favorite software, plot the decision boundary for an ANN with two inputs, two hidden units and one output. All activation functions are sigmoids. Each layer is fully connected to the next. Assume the inputs range between −5 to 5 and fix all activation thresholds to 0. Plot the decision boundaries for  the weights except the thresholds randomly chosen between (i) (−10,10), (ii) (−3,3), (iii) (−0.1,0.1) (one random set for each case is enough). Use your plots to show that weight decay can be used to control overfitting for ANNs. (If you use Matlab, the following commands might be useful: meshgrid and surf). (20 points)

Answer:

#### (i)
<img src="/written_assignments/assets/HW4/25i.png" width="400">
<img src="/written_assignments/assets/HW4/25i_surf.png" width="300">

#### (ii)
<img src="/written_assignments/assets/HW4/25ii.png" width="400">
<img src="/written_assignments/assets/HW4/25ii_surf.png" width="300">

#### (iii)
<img src="/written_assignments/assets/HW4/25iii.png" width="400">
<img src="/written_assignments/assets/HW4/25iii_surf.png" width="300">

Weight decay essentially penalizes large absolute weight magnitudes, and the surfaces above demonstrate the effects of and reasoning behind that.  As the weights decrease in allowed size, the surface, which defines the decision boundary + provides confidence information, decreases in complexity.  This is because, if we imagine each node in the netwwork as responsible for one nonlinearity, the weights determine the significance of that nonlinearity in the final model.  Smaller weights discourage large, numerous nonlinearities, which would be required for an overfit decision boundary.  As a necessary corollary, smaller weights would produce simpler models, and discourage overfitting.  

**Therefore, weight decay discourages overfitting.**

26.	When learning the weights for the perceptron, we dropped the *sign* activation function to make the objective smooth. Show that the same strategy does not work for an arbitrary ANN. (Hint: consider the shape of the decision boundary if we did this.)  (10 points)

Answer:

Removing the sign activation function from all of the hidden & output nodes of an ANN removes any possibility of nonlinearity.  Doing so ensures that, by the output node, the output is still just a linear sum of all of the input nodes in this fashion $\hat{y} = w_{21}h_{1}+w_{22}h_{2} = w_{21}(w_{11}x_{1}+w_{12}x_{2})+w_{22}(w_{13}x_{1}+w_{14}x_{2})$.  Therefore, removing the sign to calculate the gradient of the loss more easily entirely defeats the purpose of an ANN.  It would just effectively be reduced to a perceptron.
