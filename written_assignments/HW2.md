# Written Homework 2
7. Consider a learning problem where the examples are described by $n$ Boolean attributes. Prove that the number of *distinct* decision trees that can be constructed in this setting is $2^{2^n}$. *Distinct* means that each tree must represent a different hypothesis in the space. \[Hint: Show that there is a bijection between the set of all Boolean functions and the set of all distinct trees.\] (20 points)

Answer: 

Proving that the number of unique hypotheses (i.e. the number of Boolean functions) is $2^{2^n}$ is straightforward. A hypothesis maps every combination of attribute values to a predicted class label. For instance, the following table might describe a hypothesis about classifying whales:

| Fish? | Big? | Blue? | Is it a whale? |
| ----- | ----- | ----- | ----- |
|  |  |  | No |
|  |  | X | No |
|  | X | X | No |
|  | X | X | No |
| X |  |  | No |
| X |  | X | No |
| X | X |  | Yes |
| X | X |  | Yes |

All in all, there are $2^n$ possible cominations of attribute values, and we have to map each combination to one of 2 values. So, the number of unique hypotheses is $2^{number of combinations}$, or $2^{2^n}$.

Call the set of $2^{2^n}$ hypotheses $S$, and say that a decision tree 'corresponds' to a hypothesis if they have the same mapping of attribute values to class labels. We'll first show that the set of decision trees is surjective onto the set of unique hypotheses. For any hypothesis, it's possible to create a decision tree that corresponds with it. Consider a hypothesis $H$ that concerns $n$ Boolean attributes:
- Create a full binary decision tree, such that all nodes at the $i$th level partition on the $i$th attribute.
  - Each distinct path from the root to a leaf node must represent a distinct combination of attributes, as it forked from all other paths at at least one node.
  - Every possible combination of attributes must be present

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

The entropy of the original distribution is ${1 \over 2} \log{_2}{0.5} + {1 \over 2} \log{_2}{0.5} = 1$.

If you partition on $A$, both resulting 'child' partitions have their own 50-50 distribution. (The first four rows, with $A = True$, have two zeroes and two ones; the last four rows, with $A = false$, have two zeroes and two ones.) So, the entropy after partitioning on A is ${1 \over 2}({1 \over 2} \log{_2}{0.5} + {1 \over 2} \log{_2}{0.5}) + {1 \over 2}({1 \over 2} \log{_2}{0.5} + {1 \over 2} \log{_2}{0.5}) = 1$.

Similarly, partitioning on $B$ and $C$ result in 50-50 distributions, so they both have entropies of 1 as well. If partitioning on all attributes results in the same entropy as the original distribution, then all attributes have no information gain.

 
9.	Show that for a continuous attribute X, the only split values we need to check to determine a split with max IG(X) lie between points with different labels. (Hint: consider the following setting for X: there is a candidate split point $S$ in the middle of $N$ examples with the same label. To the left of $S$ are $n$ such examples. To the left of $N$, there are $L0$ examples with label negative and the rest positive, and likewise $(M0, M1)$ to the right. Express the information gain of $S$ as a function of $n$. Then show that this function is maximized either when $n=0$ or $n=N$ with all else constant.) (20 points)

Answer:

10.	Write a program to sample a set of $N$ points from $(−1,1)^2$. Label the points using the classifier $y=sign(0.5x_1+0.5x_2)$. Generate datasets from your program and use your ID3 code from Programming 1 to learn trees on this data (there is no need to do cross validation or hold out a test set). Plot a graph where the $x$-axis is the value of $N$, over $N={50, 100, 500, 1000, 5000}$, and the $y$-axis is the depth of the tree learned by ID3. Explain your observations. (20 points)

Answer: 

11.	Show the decision boundaries learned by ID3 in Q10 for $N=50$ and $N=5000$ by generating an independent test set of size 100,000, plotting all the points and coloring them according to the predicted label from the $N=50$ and $N=5000$ trees. Explain what you see relative to the true decision boundary. What does this tell you about the suitability of trees for such datasets? (20 points)

Answer:

