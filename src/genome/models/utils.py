
class EarlyStopping:
    def __init__(self, patience=1, min_delta=0, min_epochs=None):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs if min_epochs else patience
        self.counter = 0
        self.total = 0
        self.min_validation_loss = float('inf')
        self.best_model = None

    def early_stop(self, model, validation_loss):
        self.total += 1
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = model
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and self.total>self.min_epochs:
                return True
        return False
