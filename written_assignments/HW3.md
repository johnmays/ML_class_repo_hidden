# Written Homework 3

Names and github IDs (if your github ID is not your name or Case ID):

John Mays, James McHargue, Zaichuan You

12.	Under what circumstances might it be beneficial to overfit? (10 points)

Answer:Overfitting is an action where the model has a lower performance on test sets and has high performance on learning sets. In this case the model is memorizing the train set instead of learning it. \
The only circumstance which we want to overfit is when our training data is good enough. When we know that our training data set is good enough and contains all possible information that a future input may have, we will expect our model to overfit on this set of training examples. Example of this could be a library system. When you want to have a model which inputs the book and outputs the location of the book. Because there are only a limited number of books and when newer versions of books come in they will be put exactly where the old ones are. In this case memorizing the input examples would have a positive effect on future performance.


13.	Restriction biases of learning algorithms prevent overfitting by restricting the hypothesis space, while preference biases prevent overfitting by preferring simpler concepts but not necessarily restricting the hypothesis space. Discuss the pros and cons of preference vs restriction biases. (10 points)

Answer:

14.	Person X wishes to evaluate the performance of a learning algorithm on a set of $n$ examples ( $n$ large). X employs the following strategy:  Divide the $n$ examples randomly into two equal-sized disjoint sets, A and B. Then train the algorithm on A and evaluate it on B. Repeat the previous two steps for $N$ iterations ( $N$ large), then average the $N$ performance measures obtained. Is this sound empirical methodology? Explain why or why not. (10 points)

Answer: This is not sound empirical methodology. There are two major issues with Person X’s approach, particularly in comparison to N-fold cross validation: it keeps the training sets relatively small, and it doesn’t ensure independence of the testing sets. 

It’s intuitive that the performance of the network improves as the size of the training set increases; but, with this approach, the training set is always only half the size of the original dataset. There’s nothing wrong with using most of the dataset for training– in fact, N-fold cross validation generally uses a very large majority of the dataset for training every time it runs the network. So, this approach hamstrings the network’s ability to learn by unnecessarily withholding information from the training set.

Additionally, because the testing set is randomly sampled for each iteration, there’s no guarantee that the examples all appear at the same rate in the testing sets– instead, it’s virtually guaranteed that some examples will appear in the testing set way more than others. Similarly, some examples will appear in the training set way more than others. This means that the N performance measures collected during training will probably have some skew to them; this skew is random, but present, and ensures that the average performance measure isn’t a true representation of the network’s performance in the general case.

15.	Two classifiers A and B are evaluated on a sample with P positive examples and N negative examples and their ROC graphs are plotted. It is found that the ROC of A dominates that of B, i.e. for every FP rate, TP rate(A) $\geq$ TP rate(B). What is the relationship between the precision-recall graphs of A and B on the same sample? (10 points)

Answer: 

16.	Prove that an ROC graph must be monotonically increasing. (10 points)

Answer:

17.	Prove that the ROC graph of a random classifier that ignores attributes and guesses each class with equal probability is a diagonal line. (10 points)

Answer: For any example, the random classifier would have a 50% chance of classifying it as positive and a 50% chance of classifying as negative. Suppose that we give the random classifier a set of examples $S$ containing $p$ positive examples and $n$ negative examples. The number of true positives $TP$ and true negatives $TN$ are essentially determined by binomial distributions with $p$ and $n$ trials respectively, and each with success probability 0.5. The expectation of the binomial distribution is $(num trials * success prob)$, so this means that:
- The expected number of true positives $E(TP) = p/2$.
- The expected number of true negatives $E(TN) = n/2$.
- Likewise, the expected number of false negatives $E(FN) = p - E(TP) = p/2$.
- The expected number of false positives $E(FP) = n - E(FP) = n/2$.

The true positive rate $\frac{TP}{TP + FN}$ is therefore $\frac{p/2}{p/2 + p/2} = \frac{1}{2}$, and the false positive rate $\frac{FP}{FP + TN}$ is $\frac{n/2}{n/2 + n/2} = \frac{1}{2}$. The position on the ROC curve is (0.5, 0.5)-- right on the diagonal line.

If the random classifier uses confidences (i.e. assigns every example a confidence sampled from the linear distribution across [0, 1]), then the ROC position will still always be on the diagonal line. Use the same terminology from the last proof. If the positive-classification threshold for the classifier is $t \in [0, 1]$, then the same logic about binomial distributions still applies, but the success rate for $TP$ is $1 - t$, and the success rate for $TN$ is $t$. Therefore:
- The expected number of true positives $E(TP) = p * (1 - t)$.
- The expected number of true negatives $E(TN) = n * t$.
- The expected number of false negatives $E(FN) = p * t$.
- The expected number of false positives $E(FP) = n * (1 - t)$.

The true positive rate is $\frac{TP}{TP + FN} = \frac{p(1-t)}{p(1-t)+p(t)} = \frac{p(1-t)}{p} = (1-t)$

And the false positive rate is $\frac{FP}{FP + TN} = \frac{n(1-t)}{n(1-t)+n(t)} = \frac{n(1-t)}{n} = (1-t)$.

Therefore, the ROC position of this classifier is (1-t, 1-t), where $t$ is the threshold– always on the diagonal line.


