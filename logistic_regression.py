import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
plt.rcParams['figure.figsize'] = (6, 4)

class LogisticRegression(object):
    """General class for logistic regression with regularization.

    Parameters
    ----------
    learning_rate : float, optional
        Constant by which updates are multiplied (falls between 0 and 1). Defaults to 0.001.

    num_iter : int, optional
        Number of steps (or epochs) over the training data. Defaults to 10.

    penalty : None, 'L2', 'L1' or 'elasticnet'
        Option to implement regularization. Defaults to None.

    C : float, optional
        Regularization parameter; the decision function's margin. Must be positive. Defaults to 0.01.

    l1_ratio : int or float, optional
        Only for Elastic Net usage. The balance between L2 & L1 penalties (falls between 0 and 1). 
    Defaults to 0.5.
        l1_ratio = 0 is equivalent to L2 (Ridge)
        l2_ratio = 1 is equivalent to L1 (Lasso)
    """

    def __init__(self, learning_rate=0.001, num_iter=10, penalty=None, C=0.01, l1_ratio=0.5):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio

    def logistic_func(self, z):
        """Sigmoid (logistic) function, inverse of logit function.

        Parameters
        ----------
        z : float
            Linear combinations of weights and sample features.
            z = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n

        Returns
        -------
            Value of logistic function at z.
        """
        # clip z to avoid overflow in exp function
        z = np.clip(z, -500, 500)
        
        # small epsilon to avoid log(0)
        epsilon = 1e-10
        sigmoid_func = 1 / (1 + np.exp(-z))

        return np.clip(sigmoid_func, epsilon, 1 - epsilon)

    def log_likelihood(self, z, y):
        """Natural logarithm of the likelihood function (cost function to be minimized in logistic
        regression classification).

        Parameters
        ----------
        z : float
            Linear combinations of weights and sample features.
            z = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n

        y : list, shape = [n_samples,], values = 1|0
            Target values (ones or zeros).

        Returns
        -------
            Value of log-likelihood function at z and target value y.
        """
        return -1 * np.sum((y * np.log(self.logistic_func(z))) + ((1 - y) * np.log(1 - self.logistic_func(z))))

    def reg_log_likelihood(self, x, weights, y, C, penalty, l1_ratio):
        """Regularized log-likelihood function (cost function to be minimized in logistic regression
        classification with implemented reguralization).

        Parameters
        ----------
        x : {array-like}, shape = [n_samples, n_features + 1]
            Feature vectors. Note that first column of x must be a vector of ones.

        weights : {array-like}, shape = [1, 1 + n_features]
            Coefficients that weight each sample feature vector.

        y : list, shape = [n_samples,], values = 1|0
            Target values (ones or zeros).

        C : float
            Regularization parameter. C is equal to 1/lambda.

        penalty : None, 'L2', 'L1' or 'elasticnet'
            Regularization term to implement. Defaults to None.

        l1_ratio : int or float, optional
            Only for Elastic Net usage. The balance between L2 & L1 penalties (falls between 0 and 1).
            Defaults to 0.5.

        Returns
        -------
            Value of regularized log-likelihood function at z and target value y
            with regularization term (parameter).
        """
        z = np.dot(x, self.weights)
        if self.penalty == 'L2':
            reg_term = (1 / (2 * self.C)) * np.sum(self.weights**2)
        elif self.penalty == 'L1':
            reg_term = (1 / (2 * self.C)) * np.sum(np.abs(self.weights))
        elif self.penalty == 'elasticnet':
            l1_term = self.l1_ratio * np.sum(np.abs(self.weights))
            l2_term = (1 - self.l1_ratio) * np.sum(self.weights**2)
            reg_term = (1 / self.C) * (l1_term + l2_term / 2)
        else:
            reg_term = 0
            
        return self.log_likelihood(z, y) + reg_term

    def fit(self, X_train, y_train, tol=1e-4):
        """Generates coefficients (weights) to the training data.

        Parameters
        ----------
        X_train : {array-like}, shape = [n_samples, n_features]
            Training data to be fitted, where n_samples is the number of
            samples and n_features is the number of features.

        y_train : {array-like}, shape = [n_samples,], values = 1|0
            Target labels (true values).

        tol : float, optional
            Coefficient indicating the weight change between epochs in which gradient descent
            should terminated. Defaults to 0.0001

        Returns
        -------
        self : object
        """
        tolerance = tol * np.ones([1, np.shape(X_train)[1] + 1])
        self.weights = np.random.randn(np.shape(X_train)[1] + 1) * 0.01
        X_train = np.c_[np.ones([np.shape(X_train)[0], 1]), X_train]
        self.costs = []

        for i in range(self.num_iter):
            z = np.dot(X_train, self.weights)
            errors = y_train - self.logistic_func(z)

            # apply different regularizations based on the penalty
            if self.penalty == 'L2':
                delta_w = self.learning_rate * (np.dot(errors, X_train) + (1 / self.C) * self.weights)
            elif self.penalty == 'L1':
                delta_w = self.learning_rate * (np.dot(errors, X_train) + (1 / self.C) * np.sign(self.weights))
            elif self.penalty == 'elasticnet':
                l1_term = (1 / self.C) * self.l1_ratio * np.sign(self.weights)
                l2_term = (1 / self.C) * (1 - self.l1_ratio) * self.weights
                delta_w = self.learning_rate * (np.dot(errors, X_train) + l1_term + l2_term)
            else:
                delta_w = self.learning_rate * np.dot(errors, X_train)

            self.iter_performed = i

            if np.all(abs(delta_w) <= tolerance):
                break
            else:
                self.weights += delta_w                                
                if self.penalty is not None:
                    self.costs.append(self.reg_log_likelihood(X_train, self.weights, y_train, self.C, self.penalty, self.l1_ratio))
                else:
                    self.costs.append(self.log_likelihood(z, y_train))

        return self

    def predict(self, X_test, threshold=0.5):
        """Predict class label.

        Parameters
        ----------
        X_test : {array-like}, shape = [n_samples, n_features]
            Testing data, where n_samples is the number of samples and n_features is the number
            of features. n_features must be equal to the number of features in X_train.

        threshold : float, cut-off probability, optional
            Probability threshold for predicting positive class. Defaults to 0.5.

        Returns
        -------
        predictions : list, shape = [n_samples,], values = 1|0
            Class label predictions based on the weights fitted following 
            training phase.
        """
        self.threshold = threshold
        
        z = self.weights[0] + np.dot(X_test, self.weights[1:])
        probs = self.logistic_func(z)
        predictions = np.where(probs >= threshold, 1, 0)
        
        return predictions

    def predict_proba(self, X_test):
        """Predict class labels.

        Parameters
        ----------
        X_test : {array-like}, shape = [n_samples, n_features]
            Testing data, where n_samples is the number of samples and n_features is the number
            of features. n_features must be equal to the number of features in X_train.

        Returns
        -------
        probs : list, shape = [n_samples,]
            Probability that the predicted class label belongs to the positive class
            (falls between 0 and 1).
        """
        z = self.weights[0] + np.dot(X_test, self.weights[1:])
        probs = self.logistic_func(z)
        
        return probs

    def evaluate(self, predictions, y_test):
        """Computes performance metrics for binary classification.

        Parameters
        ----------
        predictions : list, shape = [n_samples,], values = 1|0
            Class label predictions based on the weights fitted following 
            training phase.
        
        y_test : list, shape = [n_samples,], values = 1|0
            True class labels.
        
        Returns
        -------
        metrics : dict
            1. accuracy,
            2. recall (sensitivity or TPR - true positive rate),
            3. precision (PPV - positive predictive value),
            4. error rate,
            5. F1 score,
            6. F2 score,
            7. specificity,
            8. arithmetric mean,
            9. geometric mean,
            10. harmonic mean,
            11. Matthews correlation coefficient (MCC),
            12. negative prediction value (NPV),
            13. false negative rate (FNR or miss rate),
            14. false positive rate (FPR or fall out),
            15. true negative rate (TNR),
            16. false discover rate (FDR),
            17. log loss,
            18. receiver operating characteristic (ROC),
            19. misclassified samples
        """
        TP = np.sum((predictions == 1) & (y_test == 1))
        TN = np.sum((predictions == 0) & (y_test == 0))
        FP = np.sum((predictions == 1) & (y_test == 0))
        FN = np.sum((predictions == 0) & (y_test == 1))
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        error_rate = (FP + FN) / (TP + TN + FP + FN) # or: 1.0 - accuracy
        F1 = (2 * TP) / (2 * TP + FP + FN)
        F2 = 5 * ((precision * recall) / (4 * precision + recall))
        specificity = TN / (TN + FP)
        arithmetric_mean = (recall + specificity) / 2
        geometric_mean = (recall * specificity)**0.5
        harmonic_mean = (2 * (recall * specificity)) / (recall + specificity)
        denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**0.5
        if denominator == 0:
            MCC = 0
        else:
            MCC = (TP * TN - FP * FN) / denominator 
        NPV = TN / (TN + FN)        
        FNR = FN / (TP + FN) # or: 1.0 - recall
        FPR = FP / (FP + TN) # or: 1.0 - specificity
        TNR = TN / (TN + FP)
        FDR = FP / (FP + TP)

        def log_loss(predictions, y_test):
            # small epsilon to avoid division by zero
            epsilon = 1e-15
            predicted_new = [max(i, epsilon) for i in predictions]
            predicted_new = [min(i, 1 - epsilon) for i in predicted_new]
            predicted_new = np.array(predicted_new)
            
            return -np.mean(y_test * np.log(predicted_new) + (1 - y_test) * np.log(1 - predicted_new))

        ROC = ((recall**2 + specificity**2)**0.5) / (2**0.5)
        missed_samples = (y_test != predictions).sum()
        
        metrics = {
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'Error Rate': error_rate,
            'F1 Score': F1,
            'F2 Score': F2,
            'Specificity': specificity,
            'Arithmetric Mean': arithmetric_mean,
            'Geometric Mean': geometric_mean,
            'Harmonic Mean': harmonic_mean,
            'Matthews Correlation Coefficient': MCC,
            'Negative Prediction Value': NPV,
            'False Negative Rate': FNR,
            'False Positive Rate': FPR,
            'True Negative Rate': TNR,
            'False Discover Rate': FDR,
            'Log Loss': log_loss(predictions, y_test),
            'Receiver Operating Characteristic': ROC,
            'Misclassified Samples': missed_samples,
        }
        
        return metrics

    def score(self, X_test, y_test):
        """Computes performance with accuracy. Implemented for scikit-learn's GridSearchCV method uasge.

        Parameters
        ----------
        X_test : {array-like}, shape = [n_samples, n_features]
            Testing data, where n_samples is the number of samples and n_features is the number
            of features. n_features must be equal to the number of features in X_train.

        y_test : list, shape = [n_samples,], values = 1|0
            True class labels.
        
        Returns
        -------
        accuracy : float
            Basic evaluation metric, the ratio of the number of correct predictions to the
            total number of input samples.
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        return accuracy

    def get_params(self, deep=True):
        """Get parameters for this estimator. Crucial for scikit-learn's GridSearchCV method uasge.

        Parameters
        ----------
        deep : boolean, optional
            Explains if model contains any other estimators or objects as attributes (like nested models),
            their parameters should also be returned. Defaults to True.
        
        Returns
        -------
            Implemented hyperparameters for grid search.
        """
        return {
            'learning_rate': self.learning_rate,
            'num_iter': self.num_iter,
            'penalty': self.penalty,
            'C': self.C,
            'l1_ratio': self.l1_ratio
        }

    def set_params(self, **params):
        """Set the parameters of this estimator. Crucial for scikit-learn's GridSearchCV method uasge.

        Parameters
        ----------
        **params : dict
            Allows the method to accept a variable number of keyword arguments (parameters) passed as
            key-value pair, where the key is the parameter name and the value is the new value to be assigned.

        Returns
        -------
        self : object
        """
        for key, value in params.items():
            setattr(self, key, value)
            
        return self
        
    def plot_prediction_curve(self, X_test, y_test):
        """Plots test samples fitted into sigmoid (logistic) function according
        to fitted weights and probabilities.

        Parameters
        ----------
        X_test : {array-like}, shape = [n_samples, n_features]
            Testing data, where n_samples is the number of samples and n_features is the number
            of features. n_features must be equal to the number of features in X_train.

        y_test : list, shape = [n_samples,], values = 1|0
            True class labels.

        Returns
        -------
        matplotlib figure
        """
        z = self.weights[0] + np.dot(X_test, self.weights[1:])
        probs = self.logistic_func(z)

        plt.plot(np.arange(-10, 10, 0.1), self.logistic_func(np.arange(-10, 10, 0.1)))
        for idx, lab in enumerate(np.unique(y_test)):
            plt.scatter(
                x=z[np.where(y_test == lab)[0]],
                y=probs[np.where(y_test == lab)[0]],
                alpha=0.8,
                marker='o',
                label=lab,
                s=30
            )

        plt.ylim([-0.1, 1.1])
        plt.axhline(0.0, ls='dotted', color='black')
        plt.axhline(self.threshold, ls='--', color='black', label='threshold')
        plt.axhline(1.0, ls='dotted', color='black')
        plt.axvline(0.0, ls='dotted', color='black')
        plt.title('Logistic Regression Prediction Curve')
        plt.xlabel(r'$z$', size='x-large')
        plt.ylabel(r'$\phi (z)$', size='x-large')
        plt.legend(loc='upper left')
        plt.show()

    def plot_decision_boundaries(self, X_test, y_test, resolution=0.01):
        """Displays decision boundaries of trained classifier. Note that this method can
        only be used when training the classifier with two features (2D plain).

        Parameters
        ----------
        X_test : {array-like}, shape = [n_samples, n_features]
            Testing data, where n_samples is the number of samples and n_features is the number
            of features. n_features must be equal to the number of features in X_train.

        y_test : list, shape = [n_samples,], values = 1|0
            True class labels.

        threshold : float, cut-off probability, optional
            Probability threshold for predicting positive class. Defaults to 0.5.

        resolution : float, cut-off probability, optional
            Resolution of contour grid.

        Returns
        -------
        matplotlib figure
        """
        x = np.arange(min(X_test[:, 0]) - 1, max(X_test[:, 0]) + 1, resolution)
        y = np.arange(min(X_test[:, 1]) - 1, max(X_test[:, 1]) + 1, resolution)        
        xx, yy = np.meshgrid(x, y, indexing='xy')

        data_points = np.transpose([xx.ravel(), yy.ravel()])
        predictions = self.predict(data_points, self.threshold)
        #probs = self.predict_proba(data_points)

        markers = ['s', '^']
        marker_colors = ['#1f77b4', '#ff7f0e']
        preds = predictions.reshape(xx.shape)
        cmap = colors.ListedColormap(marker_colors)

        # plot contours and fillings
        plt.contourf(xx, yy, preds, alpha=0.4, cmap=cmap)
        plt.contour(xx, yy, preds, linewidths=0.2, colors='black')

        # plot markers
        for idx, lab in enumerate(np.unique(y_test)):
            plt.scatter(
                x=X_test[:, 0][np.where(y_test == lab)[0]],
                y=X_test[:, 1][np.where(y_test == lab)[0]],
                alpha=0.8,
                marker=markers[idx],
                label=lab,
                s=30,
                linewidths=0.7,
                edgecolor='black',
                color=marker_colors[idx]
            )

        plt.title('Decision Boundaries')
        plt.xlabel('$x_1$', size='x-large')
        plt.ylabel('$x_2$', size='x-large')
        plt.legend(loc='best')
        plt.show()

    def plot_cost(self):
        """Plots value of natural logarithm of the likelihood cost function for each epoch.

        Returns
        -------
        matplotlib figure
        """
        plt.plot(np.arange(1, self.iter_performed + 2), self.costs, marker='.')
        plt.xticks(np.arange(1, self.iter_performed + 2))
        plt.title('Log-Likelihood Loss')
        plt.xlabel('Iterations')
        plt.ylabel(r'Log-Likelihood $J(w)$')
        plt.show()

    def plot_confusion_matrix(self, predictions, y_test, normalized=True):
        """Plots heatmap for confusion matrix.

        Parameters
        ----------
        predictions : list, shape = [n_samples,], values = 1|0
            Class label predictions based on the weights fitted following 
            training phase.
        
        y_test : list, shape = [n_samples,], values = 1|0
            True class labels.

        normalized : boolean, optional
            Whether or not to display normalized results (ranged between 0 and 1). Defaults to True.
            
        Returns
        -------
        matplotlib figure
        """
        TP = np.sum((predictions == 1) & (y_test == 1))
        TN = np.sum((predictions == 0) & (y_test == 0))
        FP = np.sum((predictions == 1) & (y_test == 0))
        FN = np.sum((predictions == 0) & (y_test == 1))

        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        normalized_matrix = confusion_matrix / confusion_matrix.sum()

        fig, ax = plt.subplots()
        if normalized:
            cax = ax.matshow(normalized_matrix, cmap='Blues')
            plt.title('Normalized Confusion Matrix')
            for (i, j), val in np.ndenumerate(normalized_matrix):
                ax.text(j, i, s=f'{val:.3f}', ha='center', va='center')
        else:
            cax = ax.matshow(confusion_matrix, cmap='Blues')
            plt.title('Confusion Matrix')
            for (i, j), val in np.ndenumerate(confusion_matrix):
                ax.text(j, i, s=f'{val}', ha='center', va='center')
            fig.colorbar(cax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    