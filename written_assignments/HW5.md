# Written Homework 5

Names and github IDs (if your github ID is not your name or Case ID):

27.	Redo the backprop example done in class (lecture 10 slide 5) with one iteration of gradient descent instead of two iterations of SGD as done in class. Compare the average losses after GD and SGD. Discuss the differences you observe in the weights and the losses. (10 points)

Answer: 

Answer 28-30 with the following scenario. The Bayesian Candy Factory makes a Halloween Candy Box that contains a mix of yummy (Y) and crummy (C) candy. You know that each Box is one of three types: 1. 80% Y and 20% C, 2. 55% Y and 45% C and 3. 30% Y and 70% C. You open a Box and start munching candies. Let the $i^{th}$ candy you munch be denoted by $c_i$. Answer the following questions using a program written in any language of your choice. Generate one Box with 100 candies for each type, and assume a fixed order of munching.
 
28.	For each Box, plot $\Pr(T=i|c_1,\ldots ,c_N)$ on a graph where $T$ represents a type and $N$ ranges from 1 to 100. (You should have three graphs and each graph will have three curves.) (10 points)

Answer:

29.	For each Box, plot $\Pr(c_{N+1}=C|c_1,\ldots ,c_N)$ where $N$ ranges from 1 to 99. (10 points)

Answer:

30.	Suppose before opening a Box you believe that each Box has 70% crummy candies (type 3) with probability 0.8 and the probability of the other two types is 0.1 each. Replot $\Pr(T=i|c_1,…,c_N)$ taking this belief into account for each of the 3 Boxes. Briefly explain the implications of your results. (10 points)

Answer: 

31.	For a constrained programming problem $\min_w f(w)$ s.t. $g_i(w) \leq 0, h_j(w)=0$, the generalized Lagrangian is defined by $L(w,\alpha,\beta)=f(w)+\sum_i \alpha_i g_i(w)+ \sum_j \beta_j h_j(w), \alpha_i \geq 0$. A primal linear program is a constrained program of the form: $\min_x c^Tx$ s.t. $Ax \geq b, x \geq 0$ where $T$ represents the transpose. Using the generalized Lagrangian, show that the dual form of the primal LP is $\max_u b^Tu$ s.t. $A^Tu \leq  c, u \geq 0$. (10 points)

Answer: We’ve established that the form of a primal linear program is $\min_x c^Tx$ s.t. $Ax \geq b, x \geq 0$. If we try to convert this to a format that the Lagrangian accepts, we get the following problem:

- $f(x) = c^T x$
- $g_i(x) = b_i - (Ax)_i$

Therefore, the generalized Lagrangian for this problem is

$\ell(w, \alpha) = c^T w + \sum_i{\alpha_i (b_i - A_i w)}$

where $A_i$ denotes the i-th row of $A$. This makes sense in the context of the *primal Lagrangian problem*, which frames the problem as $min_w max_{\alpha} \ell(w, \alpha)$. We’re predominantly trying to find the $w$ that minimizes $c^T w$. But, the Lagrangian cleverly includes the linear constraints using the $max_{\alpha}$: if $b_i - A_i w > 0$ for any constraint $i$, then the $\alpha$ that maximizes the statement would be infinity, forcing the problem to avoid that value of $w$. **This is how the problem encodes the linear constraints**, and understanding this is critical to deducing the dual linear program. 

By using the rules of matrix multiplication, we can rewrite the Lagrangian in a way that keeps the same format. We start by establishing that $\sum_i{\alpha_i (b_i - A_i w)}$ is definitionally the same as $\alpha^T (b - A w)$. Then:

- $\ell(w, \alpha) = c^T w + \alpha^T (b - A w)$
- $\ell(w, \alpha) = w^T c + \alpha^T b - \alpha^T A w$, by the distributive property of matrix multiplication
- $\ell(w, \alpha) = w^T c + b^T \alpha - w^T A^T \alpha$, as all terms in the expression are scalars (1x1 matrices) and can be transposed with no effect
- $\ell(w, \alpha) = b^T \alpha + w^T c - w^T A^T \alpha$
- $\ell(w, \alpha) = b^T \alpha + w^T (c - A^T \alpha)$, by the distributive property of matrix multiplication
- $\ell(w, \alpha) = b^T \alpha + \sum_i{w^T_{i} (c_{i} - A^T_{i} \alpha)}$

This has essentially the same format as the original Lagrangian, but it’s in a format that’s easier to interpret for the dual problem. The *dual Lagrangian problem* frames the problem as $max_{\alpha} min_w \ell(w, \alpha)$, or

$max_{\alpha} min_w (b^T \alpha + \sum_i{w^T_{i} (c_{i} - A^T_{i} \alpha)})$

**This means that the dual linear problem predominantly aims to maximize $b_T \alpha$**. But, just like the primal problem, it’s subjected to encoded constraints: if $A^T_{i} \alpha$ ever exceeds $c_{i}$ at any index $i$, then $min_w$ would be able to minimize the statement to negative infinity by setting $w_{i}$ to infinity. **So, the dual problem is also subject to the constraint that $A^T \alpha \leq c$.** This is the same as the form of the dual linear program we learned in class. (It should really go without saying that $\alpha$ is interchangeable with $u$ in this problem-- it's just a different notation.)


32.	Suppose $K_1$ and $K_2$ are two valid kernels. Show that for positive $a$ and $b$, the following are also valid kernels: (i) $aK_1+bK_2$ and (ii) $aK_1K_2$, where the product is the Hadamard product: if $K=K_1K_2$ then $K(x,y)=K_1(x,y)K_2(x,y)$. (10 points)

Answer: If $K_1$ and $K_2$ are both valid kernels, then that means there’s some valid $\Phi$ functions $\Phi_1$ and $\Phi_2$ associated with them. If we can construct a Phi function for $K$ ($\Phi_{new}$) using them, then we can show that the resulting kernel function is valid.

(i) If $K(x, y) = a K_1(x, y) + b K_2(x, y)$, then by definition of the kernel function,

$K(x, y) = a (\Phi_1(x) \cdot \Phi_1(y)) + b (\Phi_2(x) \cdot \Phi_2(y))$

$K(x, y) = \sum_i{(a \Phi_1(x)_i \Phi_1(y)_i)} + \sum_j{b (\Phi_2(x)_j \Phi_2(y)_j)}$

$K(x, y) = \sum_i{(\sqrt{a} \Phi_1(x)_i \sqrt{a} \Phi_1(y)_i)} + \sum_j{(\sqrt{b} \Phi_2(x)_j \sqrt{b} \Phi_2(y)_j)}$

Just by looking at this, it’s easy to imagine the associated $\Phi_{new}$ associated with this function:

$\Phi_{new}(x) = \sqrt{a} \Phi_1(x) \oplus \sqrt{b} \Phi_2(x)$

If you define K(x, y) as $\Phi_{new}(x) \cdot \Phi_{new}(y)$, then $K$ computes $a K_1(x, y) + b K_2(x, y)$.

(ii) This one is trickier to represent. Similar to part (i), we can say that

$K(x, y) = a (\Phi_1(x) \cdot \Phi_1(y)) (\Phi_2(x) \cdot \Phi_2(y))$

Breaking down the dot product, this equals

$K(x, y) = a (\sum_i{\Phi_{1, i}(x) \Phi_{1, i}(y)}) (\sum_j{\Phi_{2, j}(x) \Phi_{2, j}(y)})$

$K(x, y) = a (\sum_{i,j}{(\Phi_{1, i}(x) \Phi_{2, j}(x)) (\Phi_{1, i}(y) \Phi_{2, j}(y))})$

$K(x, y) = \sum_{i,j}{(\sqrt{a} \Phi_{1, i}(x) \Phi_{2, j}(x)) (\sqrt{a} \Phi_{1, i}(y) \Phi_{2, j}(y))}$

If the vector produced by $\Phi_1$ is of length $M$, and the vector produced by $\Phi_2$ is of length $N$, this corresponds to a non-linear transformation $\Phi_{new}$ that produces a vector of length $MN$:

**Because both kernels $K$ have a valid corresponding $\Phi$, both kernels are valid.**


33.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by finding $\phi$ so that $K= \phi(x)\cdot \phi(y)$. (10 points)

Answer: This can be solved similarly to part (ii) from the previous problem. So long as we can rewrite the kernel function as a sum of products between expressions in terms of $x$ and expressions in terms of $y$, we can create a corresponding $\Phi$ function for the kernel.

We can start by expanding the kernel function:

$K(x, y) = (x \cdot y + c)^2$

$K(x, y) = (x \cdot y + c) (x \cdot y + c)$

$K(x, y) = (\sum_i{x_{i} y_{i}} + c) (\sum_j{x_{j} y_{j}} + c)$

Then, by re-factoring the expression:

$K(x, y) = \sum_i{x_{i} y_{i}} \sum_j{x_{j} y_{j}} + 2 c \sum_k{x_{k} y_{k}} + c^2$

$K(x, y) = \sum_{i,j}{(x_{i} x_{j}) (y_{i} y_{j})} + \sum_k{(\sqrt{2c} x_{k}) (\sqrt{2c} y_{k})} + (c)(c)$

At this point, it becomes clear that we can create a $\Phi(x)$ function for $K$. The transformation would be a concatenation of:
- the products of all combinations of two values in $x$, with combinations with different values multiplied by $\sqrt{2}$
- the original vector $x$ scaled by $\sqrt{2c}$
- the value $c$

$\Phi(x) = [x_1^2, \sqrt{2}x_1 x_2, … \sqrt{2}x_i x_j, …, x_N^2] \oplus [\sqrt{2c} x_1, \sqrt{2c} x_2, … \sqrt{2c} x_N] \oplus [c]$


34.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by showing that it is symmetric positive semidefinite. (10 points)

Answer:

35.	Consider a modified SVM formulation derived using the plus-plane at $w\cdot x+b=c_1$ and the minus-plane at $w\cdot x+b=c_2$ , $c_1>0, c_2<0, c_1\neq −c_2$. Explain the relationship between the decision surface obtained in this case and the decision surface obtained when $c_1= −c_2$. When would we prefer one over the other? (10 points)

Answer: When $c_1 = -c_2$, then the decision boundary for the SVM will be placed exactly ‘in the middle’ between the positive and negative support vectors. That is, the decision boundary will have an equal margin on both sides between itself and the positive/negative training data. If we had $c_1 \neq -c_2$, then the decision would fall closer to one support vector, and one of the two sides would have a smaller margin. If $c_1 > -c_2$, for instance, then the positive examples would have a greater margin than the negative examples.

Which setting we prefer is determined by how we want to classify ambiguous examples. If we want ambiguous examples to be considered positive to “be safe” (e.g. testing for COVID), then we would want the margin for the plus-plane to be much larger than the margin for the minus-plane, and so we would want $c_1 > -c_2$. But, if we don’t have a preference for how ambiguous examples are classified, we wouldn’t have any reason to add this bias, and it would be preferable to keep $c_1 = -c_2$.


36.	Show with an example that an ensemble where elements have error rates worse than chance may have an overall error rate that is arbitrarily bad. (10 points)

Answer: If the error rate $\epsilon$ for the classifiers is even slightly greater than 0.5 (i.e. worse than chance), then with enough classifiers, the majority vote of the ensemble is virtually guaranteed to be incorrect. This is because the expected number of incorrect votes will be **slightly** greater than half of the number of classifiers– to be exact, with $n$ classifiers, and with $\epsilon > 0.5$, the expectation of the binomial distribution is $n \epsilon > 0.5 n$. With a small number of classifiers, the expectation may consistently be close enough to half the classifiers for the overall error rate to stay around 0.5. **But, if the number of classifiers is large, then the expected number of incorrect votes will grow far beyond half of the number of classifiers.** As such, the error rate of the ensemble can be arbitrarily bad, as the ‘bulk’ of the binomial distribution gets shifted past the majority threshold.

An example of this is illustrated in binomial.pdf. The chart plots the probability that $x$ classifiers guess incorrectly over $x$, with 999999 classifiers, given that all classifiers have an error rate of 0.501. The chart turns green at x = 500000, the point at which the incorrect votes would become the majority. It’s easy to see that virtually the entirety of the probability mass is past this point– in fact, 97.71% of the time, more than half of the classifiers will produce the wrong answer. With an error rate of 0.502, this chance becomes 99.99%. **Even with an error rate marginally worse than chance, the ensemble is essentially guaranteed to fail.**


37.	Suppose an ensemble of size 100 has two types of classifiers: $k$ “good” ones with error rates equal to 0.2 each and $m$ “bad” ones with error rates 0.6 each ( $k + m = 100$ ). Examples are classified through a majority vote. Using your favorite software/language, find a range for $k$ so that the ensemble still has an error rate < 0.5. Attach a pdf of your code to the answer.  (10 points)

Answer:

38.	Suppose a learner uses bootstrap resampling to construct a training sample T  from an initial sample U, of the same size as U. Show that, for a large enough U, the probability that some example from U appears in T is approximately 0.63. (10 points)

Answer:

39.	When boosting with $m$ examples labeled with $\pm 1$, let the weight of the $i^{th}$ example in the $t^{th}$ iteration be denoted by $W_t (i)$. Prove the following identity:
$W_{t+1} (i)$ $=exp(−y_i\sum_s \alpha_sh_s(x_i))/(m\prod_s Z_s)$.
Here $Z_s$ denotes the normalizing constant during the $s^{th}$ iteration and   $1\leq s \leq t$. (You can treat the starting point as iteration zero.) (10 points)

Answer:

All done\!
