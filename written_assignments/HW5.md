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
- $\ell(w, \alpha) = w^c + \alpha^T b - \alpha^T A w$, by the distributive property of matrix multiplication
- $\ell(w, \alpha) = w^T c + b^T \alpha - w^T A^T \alpha$, as all terms in the expression are scalars (1x1 matrices) and can be transposed with no effect
- $\ell(w, \alpha) = b^T \alpha + w^T c - w^T A^T \alpha$
- $\ell(w, \alpha) = b^T \alpha + w^T (c - A^T \alpha)$, by the distributive property of matrix multiplication
- $\ell(w, \alpha) = b^T \alpha + \sum_i{w^T_{i} (c_{i} - A^T_{i} \alpha)}$

This has essentially the same format as the original Lagrangian, but it’s in a format that’s easier to interpret for the dual problem. The *dual Lagrangian problem* frames the problem as $max_{\alpha} min_w \ell(w, \alpha)$, or

$max_{\alpha} min_w (b^T \alpha + \sum_i{w^T_{i} (c_{i} - A^T_{i} \alpha)})$

**This means that the dual linear problem predominantly aims to maximize $b_T \alpha$**. But, just like the primal problem, it’s subjected to encoded constraints: if $A^T_{i} \alpha$ ever exceeds $c_{i}$ at any index $i$, then $min_w$ would be able to minimize the statement by setting $w$ to infinity. **So, the dual problem is also subject to the constraint that $A^T \alpha \leq c$.** This is the same as the form of the dual linear program we learned in class. (It should really go without saying that $\alpha$ is interchangeable with $u$ in this problem-- it's just a different notation.)


32.	Suppose $K_1$ and $K_2$ are two valid kernels. Show that for positive $a$ and $b$, the following are also valid kernels: (i) $aK_1+bK_2$ and (ii) $aK_1K_2$, where the product is the Hadamard product: if $K=K_1K_2$ then $K(x,y)=K_1(x,y)K_2(x,y)$. (10 points)

Answer:

33.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by finding $\phi$ so that $K= \phi(x)\cdot \phi(y)$. (10 points)

Answer:

34.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by showing that it is symmetric positive semidefinite. (10 points)

Answer:

35.	Consider a modified SVM formulation derived using the plus-plane at $w\cdot x+b=c_1$ and the minus-plane at $w\cdot x+b=c_2$ , $c_1>0, c_2<0, c_1\neq −c_2$. Explain the relationship between the decision surface obtained in this case and the decision surface obtained when $c_1= −c_2$. When would we prefer one over the other? (10 points)

Answer:

36.	Show with an example that an ensemble where elements have error rates worse than chance may have an overall error rate that is arbitrarily bad. (10 points)

Answer:

37.	Suppose an ensemble of size 100 has two types of classifiers: $k$ “good” ones with error rates equal to 0.2 each and $m$ “bad” ones with error rates 0.6 each ( $k + m = 100$ ). Examples are classified through a majority vote. Using your favorite software/language, find a range for $k$ so that the ensemble still has an error rate < 0.5. Attach a pdf of your code to the answer.  (10 points)

Answer:

38.	Suppose a learner uses bootstrap resampling to construct a training sample T  from an initial sample U, of the same size as U. Show that, for a large enough U, the probability that some example from U appears in T is approximately 0.63. (10 points)

Answer:

39.	When boosting with $m$ examples labeled with $\pm 1$, let the weight of the $i^{th}$ example in the $t^{th}$ iteration be denoted by $W_t (i)$. Prove the following identity:
$W_{t+1} (i)$ $=exp(−y_i\sum_s \alpha_sh_s(x_i))/(m\prod_s Z_s)$.
Here $Z_s$ denotes the normalizing constant during the $s^{th}$ iteration and   $1\leq s \leq t$. (You can treat the starting point as iteration zero.) (10 points)

Answer:

All done\!
