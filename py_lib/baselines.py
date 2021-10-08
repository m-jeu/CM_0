import abc

import numpy as np
import pandas as pd
from scipy import stats


class DumbPredictor(metaclass=abc.ABCMeta):
    """An abstract class that represents a model that predicts new records in a simple manner based on
     past target variable values, with only simple logic with the purpose of using it as a base-line score to
     improve upon with actual models.

     Attributes:
        targets: targets from train dataset."""

    def __init__(self, target_values: pd.Series):
        """Initialize instance with targets attribute.

        Args:
            target_values: self.targets."""
        self.targets = target_values

    @abc.abstractmethod
    def _single_predict(self):
        """Make a naive 'prediction'.

        Returns:
            'prediction'."""
        pass

    def predict(self, features: np.ndarray):
        """Make a prediction for every set of features from a 1 or 2 dimensional array of features.
        Does not use actual values of features, but uses shape to determine desired size of result.

        Returns:
            1-dimensional array with 'predicted' result."""
        n_features = features.shape[0]  # Amount of feature-sets to predict label or value for.
        return np.full((n_features,), self._single_predict())  # Call _single_predict method of concrete class for value


class DumbRegressor(DumbPredictor):
    """Concrete implementation of DumbPredictor for continues target variables that 'predicts'
    a new record by always guessing past targets' mean value."""

    def _single_predict(self):
        """'Predict' a new continues value by guessing the mean value of the target variable in the training
        dataset.

        Returns:
            self.targets mean."""
        return np.mean(self.targets)


class DumbOrdinalClassifier(DumbPredictor):
    """Concrete implementation of DumbPredictor for ordinal target variables that 'predicts'
    a new record by always guessing past targets' median value."""

    def _single_predict(self):
        """Classify a new value by guessing the median value of the target variable in the training dataset.

        Returns:
            self.targets median."""
        return np.median(self.targets)


class DumbNominalClassifier(DumbPredictor):
    """Concrete implementation of DumbPredictor for nominal target variables that 'predicts'
    a new record by always guessing past targets' mode value."""

    def _single_predict(self):
        """Classify a new value by guessing the mode value of the target variable in the training dataset.
        When more then one value is considered mode, any one of them could be returned.

        Returns:
            self.targets mode."""
        modes, _ = stats.mode(self.targets)  # 'modes' is a list of modes in case of equal frequencies.
        return modes[0]                      # First one is returned because which one is picked does not matter
                                             # For the purpose of having a baseline model to compare against.
