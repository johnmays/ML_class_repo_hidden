# Written Homework 1
1. For three random variables A, B and C, show with a clear example that the statement “A is independent of B” does not imply the statement “A is independent of B given C.” 

Answer:

Here is a valid joint probability distribution for some variables, $A$, $B$, and $C$.

$C$:

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.15 | 0.05 |
| $\neg A$ | 0.05 | 0.15 |

$\neg C$:

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.10 | 0.20 |
| $\neg A$ | 0.20 | 0.10 |

Let's consider the distribution of $A$ and $B$ without consideration of $C$:

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.25 | 0.25 |
| $\neg A$ | 0.25 | 0.25 |

$P(A) = 0.5 = \frac{0.25}{0.5} = \frac{P(A,B)}{P(B)} = P(A|B) $, therefore $A$ is independent of $B$.

However, $P(A,B|C) = \frac{P(A,B,C)}{P(C)}=\frac{0.15}{0.40} = 0.375 \neq P(A|C) \cdot P(B|C) = \frac{\sum_B P(A|C)}{P(C)}\frac{\sum_A P(B|C)}{P(C)}=\frac{0.20}{0.40}\frac{0.20}{0.40}=0.25$

Because $P(A,B|C)$ is not equal to $P(A|C) \cdot P(B|C)$, $A$ is not independent of $B$ given $C$, despite the fact that $A$ is independent of $B$.  

Therefore, the implication is false.


2. Points are sampled uniformly at random from the interval $(0,1)^2$ so that they lie on the line $x+y=1$. Determine the expected squared distance between any two sampled points. 

Answer:

Since the question was asking us to find the square distance between two dots that lies on the line $x+y=1$ from interval $(0,1)^2$, we can draw a line in xy-plane which contains all possible dots.

![Figure_1](https://user-images.githubusercontent.com/89466889/188962659-a60883f3-50ba-4789-b213-4b6317803149.png)

The length of the line would be $\sqrt{1^2+1^2} = \sqrt{2}$. Now we can rotate the line to aline with x axis and we will get a line from 0 to $\sqrt{2}$. The question now turns into finding the square distance between two dots on this line; we no longer care about (x,y). Now, we first fix one of these two points and arbitarily choose to locate it at 0. **We denote this point as x**. Before we can find the expectation of square distance between this choosen point and another random point on the line, we have to know the probability density function of the other point y. That will be: 

$$
  p(y) =
    \begin{cases}
      \frac{1}{\sqrt{2}} & \text{if $y \in(0,\sqrt{2})$}\\
      0 & \text{otherwise}
    \end{cases}       
$$

The f(y) would be:
$$f(y) = (y-x)^2$$

The expectation formula: $$E(f(x)) = \sum_{x} f(x)p_{x}(x)$$
We plug in our numbers, and since we are working with continuous variables, we use an integral:

$$\frac{1}{\sqrt{2}}\int_0^\sqrt{2} (y-x)^2dy$$

Now we have a expectation with a **fixed x** and we are going to generalize it to a **random x**.
We do that by again apply the Expectation formula on the previously fixed point but this time our f(x) has changed to: $$\frac{1}{\sqrt{2}}\int_0^\sqrt{2} (y-x)^2dy$$
Plugging in numbers, we will get: $$\frac{1}{\sqrt{2}}\int_0^\sqrt{2}\frac{1}{\sqrt{2}}\int_0^\sqrt{2} (y-x)^2dydx$$
After calculation, we get the final expectation of square distance of two random points on the line which is $\frac{1}{3}$.

3. Describe two learning tasks that might be suitable for machine learning approaches. For each task, write down the goal, a possible performance measure, what examples you might get and what a suitable hypothesis space might be. Be original---don’t write about tasks discussed in class or described in the texts. Preferably select tasks from your research area (if any). Describe any aspect of the task(s) that may not fit well with the supervised learning setting and feature vector representation we have discussed. 

Answer: 

Two learning tasks that could be solved with machine learning might be 
- learning to synthesize natural-sounding syllables of human speech (James’ research), 
- and learning to determine the locations and boundaries of cells in an image (John’s research).

For determining cell locations in an image, the goal would be to generate a set of coordinates that correctly describe the locations and boundaries of cells in the image. A good performance measure would be the sum of the distances between the predicted locations of cells and the true locations. Learning examples would be images of cells annotated with the true locations/boundaries of the cells in those images.

For synthesizing speech, solving the problem would likely require a GAN, so the goal of the network would be to generate speech samples that could ‘fool’ a discriminator network. Learning examples would be real samples of speech, and a good performance measure would be the conventional formula for GAN loss,

<p align="center">$E(log(1-D(G(n)))$</p>

where G(n) is a sample generated from random noise n, and D(x) is the discriminator’s estimate that a given sample x is real. The lower this value is, the more the discriminator believes the fake speech, and the better the generator network is performing.


4. Explain in your own words: why memorization should not be considered a valid learning approach. Try to use good, intuitive examples from human learning to motivate your arguments.

Answer: 

Memorization is not learning;  there is no real induction being done in the case of memorization. Therefore, unless an entire state space is being learned, which is only feasible for extremely basic tasks, whatever 'memorizing' concept is produced will be incapable of generalizing, and thus incapable of doing well with respect to a performance measure on any new examples.  Take learning a new language as an example.  If a student is given practice sentences, is taught their meaning, then given a quiz, their performance on the quiz is highly dependent on their degree of memorization.  If they have done a great job memorizing that the phrase _"ég á hund"_ in Icelandic (which translates to _"I have a dog"_ in English), then they'll understand that example if it comes up on the quiz.  However, when tasked with any other sentence that they haven't seen, they will totally fail to comprehend its meaning unless they have been able to extract meaningful and generalizable grammar/vocabulary content from the examples in class.  If someone memorized even a big chunk of sentences in Icelandic, but could not generalize any language concepts, they would not have learned the language – their "knowledge" would be almost completely useless.

5. Explain in your own words: why tabula rasa learning is impossible. 

Answer: 

 ‘Tabula Rasa’ learning refers to learning with no preexisting limits on your hypothesis space (i.e. no inductive bias). In other words, the learning can consider any concept that decides on a viable output based on a viable input. 
 
This kind of learning is impossible for a couple reasons–- first, in most learning situations, the hypothesis space is too large to fully explore. The size of a problem’s hypothesis space varies factorially with the number of input attributes; without some kind of guidance, searching blindly through the possible hypotheses would be completely infeasible for any real-world problem. Second, even given infinite time and computational power, any network with tabula rasa would just opt to memorize the examples it’s seen. Without any kinds of restrictions, a network could consider arbitrarily complex hypotheses, and one of those always perfectly memorizes its training data with no regard for generalization. So, with tabula rasa learning, any network/agent would have no reason to generalize, and if it can’t understand the general case, it hasn’t technically ‘learned’ anything.

6. Explain in your own words: why picking a good example representation is important for learning. Try to use good, intuitive examples from human learning to motivate your arguments.

Answer: 

  A good example representation contains relatively few attributes which can significantly decrease the entropy after partitioning the dataset into two and will not run out of attributes. **This is important because, without a good representation, a functional target concept may be much harder to find or even nonexistent.** A bad example representation could have a lot of concepts which only focus on trivial properties, so it could result in a huge hypothesis space and not effectively partition examples. A huge hypothesis space will increase the cost of learning.
  A bad human learning example representation for categorizing animals could be (fur color, # legs, # eyes, # ears), e.g. “a lion is an animal with brown fur, four legs, two eyes, two ears”. In this situation, these concepts do not contain enough information to partition lions from other animals like capybaras, foxes, or marmots, and a decision tree trying to solve the problem would end up having an impure node which only contains the chance that animal would be a lion. 

