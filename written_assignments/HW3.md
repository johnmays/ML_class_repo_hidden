# Written Homework 3

Names and github IDs (if your github ID is not your name or Case ID):

John Mays, James McHargue, Zaichuan You

12.	Under what circumstances might it be beneficial to overfit? (10 points)

Answer:\
Overfitting is an action where the model has a lower performance on test sets and has high performance on learning sets. In this case the model is memorizing the train set instead of learning it. \
There are two cases in which we want that phenomenon to happen. One is when we know our test set is noisy. An example of this could be we are training a model to learn a concept which there is no historical example existed before. We have two machines which can record data. One is significantly more precise than the other. At this point we will use all the data from the precise one to train and use the data from the poor performance one to test. Because we know that the data from the poor performance one could be noisy, we don’t care about overfitting at all. At this point we want our model to learn as much as it could from the precise dataset.\
\
Another case could be when our deployment environment is largely different from the general environment. Assume we have no access to the examples from the deployment environment, then we can only use the selected data in the general environment or generate data by ourselves. In this case, our test set will come from a general environment which can have a large difference with the deployment environment. Although we may observe a lower performance on a test set, that does not impact the performance of the model on the deployment environment.


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

In order for a function $f(x)$ to be monotonically increasing, it must be entirely (across its entire domain $D$) non-decreasing.  This implies that,

$$ \forall x \in D, \frac{\mathrm{d}f(x)}{\mathrm{d}x} \geq 0 $$

ROC curves are discrete functions from the false positive rate ( $FPR$ ) to the true positive rate ( $TPR$ ) of a binary classifier.  Therefore, for an ROC curve to be monotinically increasing,

$$\frac{\Delta TPR}{\Delta FPR} \geq 0 \text{ on } FPR = [0,1] $$

This is guranteed.  

**Proof:**  

There are a fixed number of actually positive examples, $P$.  The sets of $TP$ and $FN$ are two mutually disjoint subsets of $P$ that constitute $P$ entirely.  Therefore, if $TP$ increases by a number of examples, $FP$ decreases by the same amount.  $TN$ and $FN$ have an equivalent realtionship as subsets of $N$.

Recall that $TPR = \frac{TP}{P}$ and $FPR = \frac{FP}{N}$

Assume $FPR$ is increasing... Then,  
$FPR$ is increasing $\implies FP$ is increasing $\implies$ the positive region of the classification threshold is increasing $\implies$ the set of examples classified as positive is increasing $\implies$ either $TP$ or $FP$ must increase while the other must remain at 0 $\implies$ $TP$ cannot decrease $\implies$ $\frac{TP}{P}=TPR$ cannot decrease. 

Therefore, as $FPR$ increases, $TPR$ cannot decrease, which implies that $\frac{\Delta TPR}{\Delta FPR} \geq 0 \text{ on } FPR = [0,1] $.

Therefore, ROC curves must be monotonically increasing.

17.	Prove that the ROC graph of a random classifier that ignores attributes and guesses each class with equal probability is a diagonal line. (10 points)

Answer: 

