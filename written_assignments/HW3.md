# Written Homework 3

Names and github IDs (if your github ID is not your name or Case ID):

John Mays, James McHargue, Zaichuan You

12.	Under what circumstances might it be beneficial to overfit? (10 points)

Answer:

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

Answer: 

