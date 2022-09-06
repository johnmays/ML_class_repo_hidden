# Written Homework 1
1. For three random variables A, B and C, show with a clear example that the statement “A is independent of B” does not imply the statement “A is independent of B given C.” 

Answer:

Here is a valid joint probability distribution for some variables, $A$, $B$, and $C$.

$C$

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.15 | 0.05 |
| $\not A$ | 0.05 | 0.15 |

$\neg C$

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.10 | 0.20 |
| $\neg A$ | 0.20 | 0.10 |

Let's consider the distribution of $A$ and $B$ independent of $C$:

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.25 | 0.25 |
| $\neg A$ | 0.25 | 0.25 |

$P(A) = 0.5 = P(A|B)$, therefore $A$ is independent of $B$.

However, $P(A,B|C) = \frac{P(A,B,C)}{P(C)}=\frac{0.15}{0.40} = 0.375 \neq\\ P(A|C) \cdot P(B|C) = \frac{\sum_B P(A|C)}{P(C)}\frac{\sum_A P(B|C)}{P(C)}=\frac{0.20}{0.40}\frac{0.20}{0.40}=0.25$

Because $P(A,B|C)$ is not equal to $P(A|C) \cdot P(B|C)$, $A$ is not independent of $B$ given $C$.  

Therefore, the implication is false.


2. Points are sampled uniformly at random from the interval $(0,1)^2$ so that they lie on the line $x+y=1$. Determine the expected squared distance between any two sampled points. 

Answer:

3. Describe two learning tasks that might be suitable for machine learning approaches. For each task, write down the goal, a possible performance measure, what examples you might get and what a suitable hypothesis space might be. Be original---don’t write about tasks discussed in class or described in the texts. Preferably select tasks from your research area (if any). Describe any aspect of the task(s) that may not fit well with the supervised learning setting and feature vector representation we have discussed. 

Answer:

4. Explain in your own words: why memorization should not be considered a valid learning approach. Try to use good, intuitive examples from human learning to motivate your arguments.

Answer: 

5. Explain in your own words: why tabula rasa learning is impossible. 

Answer: 

6. Explain in your own words: why picking a good example representation is important for learning. Try to use good, intuitive examples from human learning to motivate your arguments.

Answer: 
