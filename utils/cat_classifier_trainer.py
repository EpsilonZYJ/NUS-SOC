import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
from data_loader import CatDatasetLoader, visualize_dataset

class TransferLearningTrainer:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        """
        Initialize the transfer learning trainer
        
        Args:
            num_classes: Number of classes for classification
            input_shape: Input image shape
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.base_model = None
    
    def create_model(self, model_name='efficientnet', fine_tune=False, fine_tune_layers=50):
        """
        Create a transfer learning model
        
        Args:
            model_name: Name of the pre-trained model
            fine_tune: Whether to fine-tune the pre-trained layers
            fine_tune_layers: Number of layers to fine-tune from the top
            
        Returns:
            Compiled model
        """
        # Model mapping
        model_map = {
            'efficientnet': {
                'model': tf.keras.applications.EfficientNetB0,
                'input_shape': (224, 224, 3),
                'preprocess': tf.keras.applications.efficientnet.preprocess_input
            },
            'mobilenet': {
                'model': tf.keras.applications.MobileNetV2,
                'input_shape': (224, 224, 3),
                'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input
            },
            'resnet': {
                'model': tf.keras.applications.ResNet50,
                'input_shape': (224, 224, 3),
                'preprocess': tf.keras.applications.resnet50.preprocess_input
            },
            'xception': {
                'model': tf.keras.applications.Xception,
                'input_shape': (299, 299, 3),
                'preprocess': tf.keras.applications.xception.preprocess_input
            },
            'vgg16': {
                'model': tf.keras.applications.VGG16,
                'input_shape': (224, 224, 3),
                'preprocess': tf.keras.applications.vgg16.preprocess_input
            },
            'inception': {
                'model': tf.keras.applications.InceptionV3,
                'input_shape': (299, 299, 3),
                'preprocess': tf.keras.applications.inception_v3.preprocess_input
            }
        }
        
        if model_name.lower() not in model_map:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_config = model_map[model_name.lower()]
        
        # Update input shape
        self.input_shape = model_config['input_shape']
        
        # Create base model
        base_model = model_config['model'](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.base_model = base_model
        
        print(f"Created {model_name} model with {self.num_classes} classes")
        print(f"Input shape: {self.input_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        # Fine-tuning setup
        if fine_tune:
            self.setup_fine_tuning(fine_tune_layers)
        
        return model
    
    def setup_fine_tuning(self, fine_tune_layers=50):
        """
        Setup fine-tuning for the model
        
        Args:
            fine_tune_layers: Number of layers to fine-tune from the top
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        # Unfreeze the top layers of the base model
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(self.base_model.layers) - fine_tune_layers
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning enabled for top {fine_tune_layers} layers")
        print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
    
    def train(self, train_dataset, val_dataset, epochs=20, callbacks=None):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
        
        print(f"Starting training for {epochs} epochs...")
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        results = self.model.evaluate(test_dataset, verbose=0)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        
        return results
    
    def predict(self, dataset, class_names=None):
        """
        Make predictions on a dataset
        
        Args:
            dataset: Dataset to predict on
            class_names: List of class names
            
        Returns:
            Predictions and true labels
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        predictions = self.model.predict(dataset)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_labels = []
        for _, labels in dataset:
            true_labels.extend(labels.numpy())
        
        return predicted_classes, true_labels, predictions
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        import datetime
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Generate unique filename if save_path is not provided
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"training_history_{timestamp}.png"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Training history plot saved to: {save_path}")
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        true_labels: True labels
        predicted_labels: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot. If None, saves with timestamp
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"confusion_matrix_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

def compare_models(data_loader, model_names=['efficientnet', 'mobilenet', 'resnet'], epochs=10, save_path=None):
    """
    Compare multiple models
    
    Args:
        data_loader: CatDatasetLoader instance
        model_names: List of model names to compare
        epochs: Number of epochs for training
        save_path: Path to save the comparison plot. If None, saves with timestamp
        
    Returns:
        Dictionary with model results
    """
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        # Prepare data for this model
        train_ds, val_ds, _, info = data_loader.load_and_prepare_data(
            model_name=model_name,
            validation_split=0.15,
            augment_training=True
        )
        
        # Create and train model
        trainer = TransferLearningTrainer(info['num_classes'], info['input_shape'])
        model = trainer.create_model(model_name)
        
        # Train model
        history = trainer.train(train_ds, val_ds, epochs=epochs)
        
        # Evaluate model
        val_results = trainer.evaluate(val_ds)
        
        # Store results
        results[model_name] = {
            'trainer': trainer,
            'history': history,
            'val_loss': val_results[0],
            'val_accuracy': val_results[1],
            'model': model
        }
        
        print(f"{model_name} - Val Accuracy: {val_results[1]:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot validation accuracy comparison
    plt.subplot(2, 2, 1)
    for model_name, result in results.items():
        plt.plot(result['history'].history['val_accuracy'], label=model_name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot validation loss comparison
    plt.subplot(2, 2, 2)
    for model_name, result in results.items():
        plt.plot(result['history'].history['val_loss'], label=model_name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot final accuracies
    plt.subplot(2, 2, 3)
    model_names_list = list(results.keys())
    accuracies = [results[name]['val_accuracy'] for name in model_names_list]
    plt.bar(model_names_list, accuracies)
    plt.title('Final Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Plot model parameters
    plt.subplot(2, 2, 4)
    params = [results[name]['model'].count_params() for name in model_names_list]
    plt.bar(model_names_list, params)
    plt.title('Model Parameters')
    plt.ylabel('Parameters')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"model_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to: {save_path}")
    
    return results

def main():
    """
    Main training function
    """
    # Initialize data loader
    data_dir = "../data/"  # Your dataset path
    loader = CatDatasetLoader(data_dir, batch_size=32)
    
    # Choose training mode
    training_mode = "single"  # "single" or "compare"
    
    if training_mode == "single":
        # Single model training
        model_name = 'efficientnet'  # Choose: efficientnet, mobilenet, resnet, xception, vgg16, inception
        
        # Prepare data
        train_ds, val_ds, class_names, info = loader.load_and_prepare_data(
            model_name=model_name,
            validation_split=0.15,
            augment_training=True
        )
        
        print("Dataset info:", info)
        
        # Visualize some samples
        print("Visualizing dataset samples...")
        # visualize_dataset(train_ds, class_names)
        
        # Check dataset size
        train_size = tf.data.experimental.cardinality(train_ds).numpy()
        val_size = tf.data.experimental.cardinality(val_ds).numpy()
        
        print(f"Training batches: {train_size}")
        print(f"Validation batches: {val_size}")
        
        # Add data shape debugging
        print("Checking data shapes...")
        for images, labels in train_ds.take(1):
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image dtype: {images.dtype}")
            print(f"Expected shape: (batch_size, 224, 224, 3)")
            
            # If images have extra dimensions, fix them
            if len(images.shape) > 4:
                print(f"WARNING: Images have {len(images.shape)} dimensions, expected 4")
                print("Attempting to reshape...")
                # Remove extra dimensions
                while len(images.shape) > 4:
                    images = tf.squeeze(images, axis=1)
                print(f"After reshaping: {images.shape}")
            break
        
        # Create and train model
        trainer = TransferLearningTrainer(info['num_classes'], info['input_shape'])
        model = trainer.create_model(model_name, fine_tune=True, fine_tune_layers=50)
        
        # Train model
        print("\nStarting training...")
        history = trainer.train(train_ds, val_ds, epochs=20)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        print("\nEvaluating model...")
        results = trainer.evaluate(val_ds)
        
        # Make predictions and show confusion matrix
        predicted_classes, true_labels, predictions = trainer.predict(val_ds, class_names)
        plot_confusion_matrix(true_labels, predicted_classes, class_names)
        
        # Save model
        trainer.save_model(f'cat_classifier_{model_name}.h5')
        
    elif training_mode == "compare":
        # Compare multiple models
        print("Comparing multiple models...")
        models_to_compare = ['efficientnet', 'mobilenet', 'resnet']
        
        results = compare_models(loader, models_to_compare, epochs=15)
        
        # Print comparison results
        print("\n" + "="*50)
        print("MODEL COMPARISON RESULTS")
        print("="*50)
        
        for model_name, result in results.items():
            print(f"{model_name.upper()}: Accuracy = {result['val_accuracy']:.4f}, Loss = {result['val_loss']:.4f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
        print(f"\nBest model: {best_model.upper()} with accuracy: {results[best_model]['val_accuracy']:.4f}")
        
        # Save best model
        results[best_model]['trainer'].save_model(f'best_cat_classifier_{best_model}.h5')
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
