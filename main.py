# main.py
import os
import numpy as np
from data_loader import DataLoader
from model import BrainTumorModel
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from plotter import Plotter
from utils import Utils

def main():
    # Set data directories
    data_dir = '/kaggle/input/brain-mri-images-for-brain-tumor-detection'
    augmented_data_path = '/kaggle/working/augmented-images'

    # Initialize DataLoader and create augmented directories
    data_loader = DataLoader(data_dir, augmented_data_path)
    data_loader.create_augmented_directories()

    # Augment images if not already done
    if len(os.listdir(f"{augmented_data_path}/yes")) == 0:
        data_loader.augment_images(file_dir=data_dir+'/yes', no_samples_gen=6, save_img_dir=augmented_data_path+'/yes')
        data_loader.augment_images(file_dir=data_dir+'/no', no_samples_gen=9, save_img_dir=augmented_data_path+'/no')

    # Load and preprocess data
    X_train, y_train = data_loader.load_data([f"{augmented_data_path}/yes", f"{augmented_data_path}/no"])
    X_train = [Utils.crop_brain_cnt(img) for img in X_train]
    X_train = Utils.resize_images(X_train)

    # Split data into training, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.split_data(X_train, y_train)

    # Initialize and build the model
    brain_tumor_model = BrainTumorModel(input_shape=(240, 240, 3))
    model = brain_tumor_model.get_model()

    # Train the model
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=15)

    # Evaluate the model
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X_test, y_test)

    # Plot training history
    Plotter.plot_metrics(history)

if __name__ == '__main__':
    main()
