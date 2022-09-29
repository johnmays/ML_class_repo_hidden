# Written Homework 2
7. Consider a learning problem where the examples are described by $n$ Boolean attributes. Prove that the number of *distinct* decision trees that can be constructed in this setting is $2^{2^n}$. *Distinct* means that each tree must represent a different hypothesis in the space. \[Hint: Show that there is a bijection between the set of all Boolean functions and the set of all distinct trees.\] (20 points)

Answer: 

Proving that the number of unique hypotheses (i.e. the number of Boolean functions) is $2^{2^n}$ is straightforward. A hypothesis maps every combination of attribute values to a predicted class label. For instance, the following table might describe a hypothesis about classifying whales:

| Fish? | Big? | Blue? | Is it a whale? |
| ----- | ----- | ----- | ----- |
|  |  |  | No |
|  |  | X | No |
|  | X |  | No |
|  | X | X | No |
| X |  |  | No |
| X |  | X | No |
| X | X |  | Yes |
| X | X | X | Yes |

All in all, there are $2^n$ possible cominations of attribute values, and we have to map each combination to one of 2 values. So, the number of unique hypotheses is $2^{number of combinations}$, or $2^{2^n}$.

Call the set of $2^{2^n}$ hypotheses $S$, and say that a decision tree 'corresponds' to a hypothesis if they have the same mapping of attribute values to class labels. We'll first show that the set of distinct decision trees is surjective onto the set of unique hypotheses. For any hypothesis, it's possible to create a decision tree that corresponds with it. Consider a hypothesis $H$ that concerns $n$ Boolean attributes:
- Create a full binary decision tree of depth $n$, such that all nodes at the $i^{th}$ level partition between True and False on the $i^{th}$ attribute.
  - Every possible combination of attributes must be present; by the construction of the tree, no matter what combination of values (branches) you choose, there is a matching path to a leaf node.
- Each leaf node represents a unique combination of attribute values. So, for each leaf node, classify examples based on the matching combination of values according to the hypothesis.

This tree would map every combination of attributes to a class label in perfect agreement with $H$. Since, for every hypothesis, there is a corresponding distinct decision tree that represents it, the set of distinct decision trees is surjective onto the set of hypotheses.

The set of distinct decision trees is also injective onto the set of unique hypotheses. This is a relatively trivial point to make-- if two decision trees both mapped to the same hypothesis, then by definition of 'distinct', one of them could be considered a 'non-distinct' copy of the other. Therefore, there can definitionally be only one distinct decision tree for each hypothesis.

Since the set of distinct decision trees is both surjective and injective onto the set of hypotheses (i.e. Boolean functions), the two sets must have the same size; since the size of the set of unique hypotheses is $2^{2^n}$, the size of the set of distinct decision trees must also be $2^{2^n}$.

8.	(i) Give an example of a nontrivial (nonconstant) Boolean function over $3$ Boolean attributes where IG(X) would return zero for *all* attributes at the root. (ii) Explain the significance of this observation, given your answer to Q7. (iii) Estimate how many such functions could exist over $n$ attributes, as a function of $n$. (20 points)

Answer:
(i) An example of one such indecisive Boolean function is:
| Condition | $y$ |
| ----- | ----- |
| $A$ & $B$ & $C$ | 0 |
| $A$ & $B$ & $\neg C$ | 1 |
| $A$ & $\neg B$ & $C$ | 1 |
| $A$ & $\neg B$ & $\neg C$ | 0 |
| $\neg A$ & $B$ & $C$ | 0 |
| $\neg A$ & $B$ & $\neg C$ | 1 |
| $\neg A$ & $\neg B$ & $C$ | 1 |
| $\neg A$ & $\neg B$ & $\neg C$ | 0 |

The entropy of the original distribution is $-{1 \over 2} \log{_2}{0.5} + -{1 \over 2} \log{_2}{0.5} = 1$.

If you partition on $A$, both resulting 'child' partitions have their own 50-50 distribution. (The first four rows, with $A = True$, have two zeroes and two ones; the last four rows, with $A = false$, have two zeroes and two ones.) So, the entropy after partitioning on A is ${1 \over 2}(-{1 \over 2} \log{_2}{0.5} + -{1 \over 2} \log{_2}{0.5}) + {1 \over 2}(-{1 \over 2} \log{_2}{0.5} + -{1 \over 2} \log{_2}{0.5}) = 1$.

Similarly, partitioning on $B$ and $C$ result in 50-50 distributions, so they both have entropies of 1 as well. If partitioning on all attributes results in the same entropy as the original distribution, then all attributes have no information gain.

(ii) If IG(X) is 0 for all attributes, this means that the decision tree corresponding to that Boolean function would have no information gain from partitioning in its root node. No matter what attribute you partition on at the root, the corresponding left and right subtrees always have identical class distributions to the original distribution.

(iii) Finding the exact number of such 'indecisive' functions for an arbitrary $n$ is currently an unsolved problem. But, it's possible to describe bounds on the number. Assume that the example distribution is equal across combinations of attributes; we can show that, at a minimum, every Boolean function for which $X$ is always classified with the same label as $\neg X$ is indecisive:

- Consider an arbitrary attribute $A$. If we partition on $A$, then we produce two sets of examples: $P_0$, where $A$ is false, and $P_1$, where $A$ is true.
- Because there's an equal distribution across all combinations, there are an equal number of examples for any given combination of attributes. The number of combinations for which $A$ is false is equal to the number of combinations for which $A$ is true, so the number of examples with each value is also equal, and $P_0$ and $P_1$ are the same size.
- Every example $E$ in $P_0$ has an 'inverse' combination in which every value is the opposite. By the equal distribution assumption, the number of 'inverses' of $E$ is the same as the number of instances of $E$. And trivially, these inverses must be in $P_1$.
- It's intuitive that the examples in $P_0$ would map bijectively to their respective inverses in $P_1$. So, if every example and its inverse has the same class label, then $P_0$ and $P_1$ must have identical proportions of every class label.
- If each partition has the same probabilities of class labels, and each partition is the same size, then the original example set must have also had the same probabilities of class labels. 

If the probabilities of the class labels in the original example set is the same as the probabilities of the class labels in $P_0$ and $P_1$, then the entropy of the original set, the entropy of $P_0$, and the entropy of $P_1$ must be the same, since entropy depends only on the distribution of probabilities across classes. Thus, $IG(X) = 0$ for any attribute $X$ at the root.

The number of Boolean functions where every example is classified under the same label as its inverse is $2^{2^{n-1}}$. This is because you're basically only setting labels for half of the attribute combinations, or ${2^{n} \over 2} = 2^{n-1}$ attribute combinations; the other half of the labels is just based on their inverses in the first half. **So, $\Omega(n) = 2^{2^{n-1}}$.**

A simple, but clear upper bound on the number of indecisive functions is $O(n) = 2^{2^{n} - 1}$. This is just because, for the two partitions from the root to have the same number of examples with $y = 1$, *the number of values with* $y = 1$ *must be even.* The set of Boolean functions with an even number of combinations for which $y = 1$ accounts for half of all functions. **So, an upper bound is $2^{2^n} \over 2$, or $2^{2^{n} - 1}$.**

9.	Show that for a continuous attribute X, the only split values we need to check to determine a split with max IG(X) lie between points with different labels. (Hint: consider the following setting for X: there is a candidate split point $S$ in the middle of $N$ examples with the same label. To the left of $S$ are $n$ such examples. To the left of $N$, there are $L0$ examples with label negative and the rest positive, and likewise $(M0, M1)$ to the right. Express the information gain of $S$ as a function of $n$. Then show that this function is maximized either when $n=0$ or $n=N$ with all else constant.) (20 points)

Answer:

Suppose we have a set of examples whcih have a continuous attribute X. There is a candidate split point $S$ in the middle of **N** examples which all have the same label. There are **n** examples inside the N examples set which is on the left of split point S. Correspondingly there will be **N-n** examples on the right of split point S inside the N examples set. There are **O<sub>L,0</sub>** examples on the left of the N examples which were labeled 0. There are **O<sub>L,1</sub>** examples on the left of the N examples which were labeled 1. Likewise **O<sub>M,0</sub>** and **O<sub>M,1</sub>** represents the examples on the right of N examples. **AllL** denote all the examples on the left of N. **AllM** denote all the examples on the right of N. **All example** denote all the examples in the dataset.

With these notations we can first come up with $H(Y|X=0)$ which is the entropy of examples on the left after partition.

$$
  \begin{align}
    H(Y|X=0) &= -(\frac{n+ O_{L,1}}{All L+n} log_{2}\frac{n+ O_{L,1}}{All L+n}+\frac{O_{L,0}}{All L+n} log_{2}\frac{O_{L,0}}{All L+n})
  \end{align}
$$

Similally for $H(Y|X=1)$ which is the entropy of examples on the right after partition.

$$
  \begin{align}
    H(Y|X=1) &= -(\frac{N-n+ O_{M,1}}{All M+N-n} log_{2}\frac{N-n+ O_{M,1}}{All M+N-n}+\frac{O_{M,0}}{All M+N-n} log_{2}\frac{O_{M,0}}{All M+N-n})
  \end{align}
$$

Then we can combine $H(Y|X=0)$ and $H(Y|X=1)$ to get $H(Y|X)$.

$$
  \begin{align}
    H(Y|X) &= P(X=0)H(Y|X=0)+P(X=1)H(Y|X=1)\\
    &=-\frac{AllL+n}{All\ example}(\frac{n+ O_{L,1}}{All L+n} log_{2}\frac{n+ O_{L,1}}{All L+n}+\frac{O_{L,0}}{All L+n} log_{2}\frac{O_{L,0}}{All L+n})-\frac{AllM+N-n}{All\ example}(\frac{N-n+ O_{M,1}}{All M+N-n} log_{2}\frac{N-n+ O_{M,1}}{All M+N-n}+\frac{O_{M,0}}{All M+N-n} log_{2}\frac{O_{M,0}}{All M+N-n})\\
    &=-\frac{1}{All\ example}((n+ O_{L,1}) log_{2}\frac{n+ O_{L,1}}{All L+n}+(O_{L,0}) log_{2}\frac{O_{L,0}}{All L+n})-\frac{1}{All\ example}((N-n+ O_{M,1}) log_{2}\frac{N-n+ O_{M,1}}{All M+N-n}+O_{M,0} log_{2}\frac{O_{M,0}}{All M+N-n})\\
    & \text{Since all example is a constant we can just get ride of the coefficient}\\
    &=-((n+ O_{L,1}) log_{2}\frac{n+ O_{L,1}}{All L+n}+(O_{L,0}) log_{2}\frac{O_{L,0}}{All L+n})-((N-n+ O_{M,1}) log_{2}\frac{N-n+ O_{M,1}}{All M+N-n}+O_{M,0} log_{2}\frac{O_{M,0}}{All M+N-n})\\
    & \text{Use the proprity of log}\\
    &=-[(n+ O_{L,1})log_{2}(n+ O_{L,1})-(n+ O_{L,1})log_{2}(All L+n)+ O_{L,0}log_{2}(O_{L,0})- O_{L,0}log_{2}(All L+n)]-[(N-n+ O_{M,1})log_{2}(N-n+ O_{M,1})-(N-n+ O_{M,1})log_{2}(All M+N-n)+ O_{M,0}(log_{2}(O_{M,0})-log_{2}(All M+N-n))]
  \end{align}
$$


$$
  \begin{align}
  \frac{dH(Y|X)}{dn} &= -[log_{2}(n+ O_{L,1})+\frac{n+ O_{L,1}}{ln(2)(n+ O_{L,1})}-log_{2}(All L+n)-\frac{n+ O_{L,1}}{ln(2)(All L+n)}+\frac{O_{L,0}}{ln(2)(All L+n)}]-[-log_{2}(N-n+ O_{M,1})-\frac{N-n+ O_{M,1}}{ln(2)(N-n+ O_{M,1})}+log_{2}(All M+N-n)+\frac{N-n+ O_{M,1}}{ln(2)(All M+N-n)}+\frac{O_{M,0}}{ln(2)(All M+N-n)}]\\
  &=-[log_{2}(n+ O_{L,1})-log_{2}(All L+n)-\frac{n+ O_{L,1}}{ln(2)(All L+n)}+\frac{O_{L,0}}{ln(2)(All L+n)}]+[log_{2}(N-n+ O_{M,1})-log_{2}(All M+N-n)-\frac{N-n+ O_{M,1}}{ln(2)(All M+N-n)}-\frac{O_{M,0}}{ln(2)(All M+N-n)}]\\
  &=-[log_{2}(n+ O_{L,1})-log_{2}(All L+n)-\frac{n+ O_{L,1}+O_{L,0}}{ln(2)(All L+n)}]+[log_{2}(N-n+ O_{M,1})-log_{2}(All M+N-n)-\frac{N-n+ O_{M,1}+O_{M,0}}{ln(2)(All M+N-n)}]\\
  & \text{remember how we defined O<sub>L,0</sub>, O<sub>L,1</sub>, O<sub>M,0</sub>, and O<sub>M,0</sub>. Adding them togather is just All L and All M}\\
  &=-[log_{2}(n+ O_{L,1})-log_{2}(All L+n)-\frac{1}{ln(2)}]+[log_{2}(N-n+ O_{M,1})-log_{2}(All M+N-n)-\frac{1}{ln(2)}]\\
  &=-log_{2}(n+ O_{L,1})+log_{2}(All L+n)+log_{2}(N-n+ O_{M,1})-log_{2}(All M+N-n)\\
  \end{align}
$$

$$
  \begin{align}
  \frac{dH(Y|X)^2}{dn^2} &= -\frac{1}{ln(2)(n+ O_{L,1})}+\frac{1}{ln(2)(All L+n)}-\frac{1}{ln(2)(N-n+ O_{M,1})}+\frac{1}{ln(2)(All M+N-n)}
  \end{align}
$$

From our definition of variables, we can tell that the second derivative of dH(Y|X) is smaller or equals to 0. Because n+ O<sub>L,1</sub> is always smaller or equals to All L+n and N-n+ O<sub>M,1</sub> is always smaller or equals to All M+N-n. A bigger denominator will have smaller value. So the second derivative will always be less than or equals to 0. Since H(Y) is a constant and will becocme 0 after derivative, this is also the second derivative of IG(X). So the optimal I found in first derivative will be local maxima of IG(X).

The local optima of bounded range will only exist inside the range or on the boundry. Since I can not find a point inside the range where the first derivative equals to 0, I can say that the local optima exists on the boundry which is n=0 and n=N.

10.	Write a program to sample a set of $N$ points from $(−1,1)^2$. Label the points using the classifier $y=sign(0.5x_1+0.5x_2)$. Generate datasets from your program and use your ID3 code from Programming 1 to learn trees on this data (there is no need to do cross validation or hold out a test set). Plot a graph where the $x$-axis is the value of $N$, over $N={50, 100, 500, 1000, 5000}$, and the $y$-axis is the depth of the tree learned by ID3. Explain your observations. (20 points)

Answer: 

11.	Show the decision boundaries learned by ID3 in Q10 for $N=50$ and $N=5000$ by generating an independent test set of size 100,000, plotting all the points and coloring them according to the predicted label from the $N=50$ and $N=5000$ trees. Explain what you see relative to the true decision boundary. What does this tell you about the suitability of trees for such datasets? (20 points)

Answer:

