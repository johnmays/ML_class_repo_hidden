# Written Homework 2
7. Consider a learning problem where the examples are described by $n$ Boolean attributes. Prove that the number of *distinct* decision trees that can be constructed in this setting is $2^{2^n}$. *Distinct* means that each tree must represent a different hypothesis in the space. \[Hint: Show that there is a bijection between the set of all Boolean functions and the set of all distinct trees.\] (20 points)

Answer: 

8.	(i) Give an example of a nontrivial (nonconstant) Boolean function over $3$ Boolean attributes where IG(X) would return zero for *all* attributes at the root. (ii) Explain the significance of this observation, given your answer to Q7. (iii) Estimate how many such functions could exist over $n$ attributes, as a function of $n$. (20 points)

Answer:
 
9.	Show that for a continuous attribute X, the only split values we need to check to determine a split with max IG(X) lie between points with different labels. (Hint: consider the following setting for X: there is a candidate split point $S$ in the middle of $N$ examples with the same label. To the left of $S$ are $n$ such examples. To the left of $N$, there are $L0$ examples with label negative and the rest positive, and likewise $(M0, M1)$ to the right. Express the information gain of $S$ as a function of $n$. Then show that this function is maximized either when $n=0$ or $n=N$ with all else constant.) (20 points)

Answer:

10.	Write a program to sample a set of $N$ points from $(−1,1)^2$. Label the points using the classifier $y=sign(0.5x_1+0.5x_2)$. Generate datasets from your program and use your ID3 code from Programming 1 to learn trees on this data (there is no need to do cross validation or hold out a test set). Plot a graph where the $x$-axis is the value of $N$, over $N={50, 100, 500, 1000, 5000}$, and the $y$-axis is the depth of the tree learned by ID3. Explain your observations. (20 points)

Answer: 

11.	Show the decision boundaries learned by ID3 in Q10 for $N=50$ and $N=5000$ by generating an independent test set of size 100,000, plotting all the points and coloring them according to the predicted label from the $N=50$ and $N=5000$ trees. Explain what you see relative to the true decision boundary. What does this tell you about the suitability of trees for such datasets? (20 points)

Answer:

