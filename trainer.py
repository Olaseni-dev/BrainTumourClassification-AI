# trainer.py
from tensorflow.keras.callbacks import EarlyStopping

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, X_val, y_val, epochs=15):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        history = self.model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
        return history
