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

Let's consider the distribution of $A$ and $B$ independent of $C$:

| | $B$ | $\neg B$ |
| ----- | ----- | ---- |
| $A$ | 0.25 | 0.25 |
| $\neg A$ | 0.25 | 0.25 |

$P(A) = 0.5 = \frac{0.25}{0.5} = \frac{P(A,B)}{P(B)} = P(A|B) $, therefore $A$ is independent of $B$.

However, $P(A,B|C) = \frac{P(A,B,C)}{P(C)}=\frac{0.15}{0.40} = 0.375 \neq P(A|C) \cdot P(B|C) = \frac{\sum_B P(A|C)}{P(C)}\frac{\sum_A P(B|C)}{P(C)}=\frac{0.20}{0.40}\frac{0.20}{0.40}=0.25$

Because $P(A,B|C)$ is not equal to $P(A|C) \cdot P(B|C)$, $A$ is not independent of $B$ given $C$.  

Therefore, the implication is false.


2. Points are sampled uniformly at random from the interval $(0,1)^2$ so that they lie on the line $x+y=1$. Determine the expected squared distance between any two sampled points. 

Answer:

3. Describe two learning tasks that might be suitable for machine learning approaches. For each task, write down the goal, a possible performance measure, what examples you might get and what a suitable hypothesis space might be. Be original---don’t write about tasks discussed in class or described in the texts. Preferably select tasks from your research area (if any). Describe any aspect of the task(s) that may not fit well with the supervised learning setting and feature vector representation we have discussed. 

Answer: 

Two learning tasks that could be solved with machine learning might be 
- learning to synthesize natural-sounding syllables of human speech (James’ research), 
- and learning to determine the locations of cells in an image (John’s research).

For determining cell locations in an image, the goal would be to generate values that correctly describe the locations of cells. A good performance measure would be the sum of the distances between the predicted locations of cells and the true locations. Learning examples would be images of cells annotated with the true locations of the cells in those images.

For synthesizing speech, solving the problem would likely require a GAN, so the goal of the network would be to generate speech samples that could ‘fool’ a discriminator network. Learning examples would be real samples of speech, and a good performance measure would be the conventional formula for GAN loss,

<p align="center">$E(log(1-D(G(n)))$</p>

where G(n) is a sample generated from random noise n, and D(x) is the discriminator’s estimate that a given sample x is real. The lower this value is, the more the discriminator believes the fake speech, and the better the generator network is performing.


4. Explain in your own words: why memorization should not be considered a valid learning approach. Try to use good, intuitive examples from human learning to motivate your arguments.

Answer: 

Memorization is not learning.  There is no real induction being done in the case of memorization, therefore, unless an entire state space is being learned, which is only feasible for extremely basic tasks, whatever concept is being produced will be incapable of generalizing, and thus incapable of doing well with respect to a performance measure on any new examples.  Take learning a new language as an example.  If a student is given practice sentences, is taught their meaning, then given a quiz, their performance on the quiz is highly dependent on their degree of memorization.  If they have done a great job memorizing that the phrase _"ég á hund"_ in Icelandic translates to _"I have a dog"_ in English, then they'll know that example if it comes up on the quiz.  However, when tasked with any other sentence that they haven't seen, they will totally fail to comprehend its meaning unless they have been able to extract meaningful and generalizable grammar/vocabulary content from the examples in class.  If someone memorized even a big chunk of sentences in Icelandic, but could not generalize any language concepts, they would not have learned the language – their "knowledge" would be almost completely useless.

5. Explain in your own words: why tabula rasa learning is impossible. 

Answer: 

 ‘Tabula Rasa’ learning refers to learning with no preexisting limits on your hypothesis space (i.e. no inductive bias). In other words, the learning can consider any concept that decides on a viable output based on a viable input. 
 
This kind of learning is impossible for a couple reasons– first, in most learning situations, the hypothesis space is too large to fully explore. The size of a problem’s hypothesis space varies factorially with the number of input attributes; without some kind of guidance, searching blindly through the possible hypotheses for any real-world problem would be completely infeasible. Second, even given infinite time and computational power, any network without tabula rasa would just opt to memorize the examples it’s seen. Without any kinds of restrictions, a network could consider arbitrarily complex hypotheses, and one of those always perfectly memorizes its training data with no regard for generalization. So, with tabula rasa learning, any network/agent would have no reason to generalize, and if it can’t understand the general case, it hasn’t technically ‘learned’ anything.

6. Explain in your own words: why picking a good example representation is important for learning. Try to use good, intuitive examples from human learning to motivate your arguments.

Answer: 
