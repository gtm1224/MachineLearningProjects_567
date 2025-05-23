import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Part 2: Implementation of AdaBoost with decision trees as weak learners

class AdaBoost:
  def __init__(self, n_estimators=60, max_depth=10):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.betas = []
    self.models = []
    
  def fit(self, X, y):
    ###########################TODO#############################################
    # In this part, please implement the adaboost fitting process based on the 
    # lecture and update self.betas and self.models, using decision trees with 
    # the given max_depth as weak learners

    # Inputs: X, y are the training examples and corresponding (binary) labels
    
    # Hint 1: remember to convert labels from {0,1} to {-1,1}
    # Hint 2: DecisionTreeClassifier supports fitting with a weighted training set
    y = np.where(y == 0, -1, 1)
    n_samples = X.shape[0]
    D = np.ones(n_samples) / n_samples

    for _ in range(self.n_estimators):
        clf = DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(X, y, sample_weight=D)
        pred = clf.predict(X)
        error = np.sum(D * (pred != y)) / np.sum(D)

        if error >= 1.0:
            continue
        if error == 0:
            beta = 0.1*10**11
        else:
            beta = 0.5 * np.log((1 - error) / error)

        self.models.append(clf)
        self.betas.append(beta)

        D = D * np.exp(-beta * y * pred)
        D = D / np.sum(D)

    return self
    
  def predict(self, X):
    ###########################TODO#############################################
    # In this part, make prediction on X using the learned ensemble
    # Note that the prediction needs to be binary, that is, 0 or 1.
    final_score = np.zeros(X.shape[0])
    for beta, model in zip(self.betas, self.models):
      pred = model.predict(X)
      final_score += beta * pred

    preds = (np.sign(final_score) > 0).astype(int)
    return preds
    
  def score(self, X, y):
    accuracy = accuracy_score(y, self.predict(X))
    return accuracy

