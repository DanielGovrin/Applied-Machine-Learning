from sklearn.ensemble import HistGradientBoostingClassifier


class Model:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.09816381586885223,
            max_depth=48
        )

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


