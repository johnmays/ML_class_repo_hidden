def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
    """
    This is the method where the training algorithm will run.

    Args:
        X: The dataset. The shape is (n_examples, n_features).
        y: The labels. The shape is (n_examples,)
        weights: Weights for each example. Will become relevant later in the course, ignore for now.
    """

    # In Java, it is best practice to LBYL (Look Before You Leap), i.e. check to see if code will throw an exception
    # BEFORE running it. In Python, the dominant paradigm is EAFP (Easier to Ask Forgiveness than Permission), where
    # try/except blocks (like try/catch blocks) are commonly used to catch expected exceptions and deal with them.
    try:
        split_criterion = self._determine_split_criterion(X, y)
    except NotImplementedError:
        warnings.warn('This is for demonstration purposes only.')

    n_zero, n_one = util.count_label_occurrences(y)

    if n_one > n_zero:
        self._majority_label = 1
    else:
        self._majority_label = 0