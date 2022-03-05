class NaiveClassifier:
    def __init__(self):
        self.time_mean = 0

    def train(self, train_numerical):
        self.time_mean = train_numerical[:, 0].mean()

    def forward(self, numerical):
        return (numerical[0] > self.time_mean).int()
