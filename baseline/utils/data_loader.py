import tensorflow as tf
import os
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import datetime

class CatDatasetLoader:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            data_dir: Root directory path of the dataset
            img_size: Target image size (height, width)
            batch_size: Batch size
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['Ragdolls', 'Singapura_cats', 'Persian_cats', 'Sphynx_cats', 'Pallas_cats']
        
        # Raspberry Pi optimization settings
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    
    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess a single image
        
        Args:
            image_path: Path to the image
            target_size: Target size, defaults to self.img_size
        
        Returns:
            Preprocessed image tensor
        """
        if target_size is None:
            target_size = self.img_size
            
        # Read image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize image while maintaining aspect ratio
        image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        return image
    
    def create_dataset_from_directory(self, validation_split=0.15):
        """
        Create dataset from directory structure
        
        Directory structure should be:
        data_dir/
        ├── Ragdolls/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── Singapura/
        └── ...
        
        Args:
            validation_split: Validation set ratio
            
        Returns:
            train_dataset, val_dataset, class_names
        """
        # Use tf.keras.utils.image_dataset_from_directory to create dataset
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
            class_names=self.class_names
        )
        
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
            class_names=self.class_names
        )
        
        return train_dataset, val_dataset, self.class_names
    
    def create_dataset_from_file_list(self, image_paths, labels):
        """
        Create dataset from file path list
        
        Args:
            image_paths: List of image paths
            labels: Corresponding label list
            
        Returns:
            TensorFlow Dataset
        """
        def preprocess_path_and_label(path, label):
            image = self.preprocess_image(path)
            return image, label
        
        # Create dataset from paths and labels
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((path_ds, label_ds))
        
        # Apply preprocessing
        dataset = dataset.map(
            preprocess_path_and_label,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
    
    def optimize_dataset_for_raspberry_pi(self, dataset):
        """
        Optimize dataset performance for Raspberry Pi
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Optimized dataset
        """
        # Cache dataset to memory (if memory is sufficient)
        dataset = dataset.cache()
        
        # Shuffle data (only if not already batched)
        dataset = dataset.shuffle(buffer_size=1000)
        
        # Note: Batch processing is done in create_dataset_from_directory
        # Don't batch again here to avoid double batching
        
        # Prefetch data to improve performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def apply_data_augmentation(self, dataset, augment=True):
        """
        Apply data augmentation
        
        Args:
            dataset: Input dataset
            augment: Whether to apply augmentation
            
        Returns:
            Augmented dataset
        """
        if not augment:
            # Only apply normalization
            normalization_layer = tf.keras.utils.Rescaling(1./255)
            dataset = dataset.map(
                lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            return dataset
        
        # Define data augmentation layers
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        
        # Normalization layer
        normalization_layer = tf.keras.utils.Rescaling(1./255)
        
        def augment_and_normalize(image, label):
            # Apply normalization first
            image = normalization_layer(image)
            # Apply augmentation
            image = data_augmentation(image, training=True)
            return image, label
        
        dataset = dataset.map(
            augment_and_normalize,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
    
    def get_model_specific_preprocessing(self, model_name):
        """
        Get model-specific preprocessing function
        
        Args:
            model_name: Model name ('efficientnet', 'mobilenet', 'resnet', 'xception')
            
        Returns:
            Preprocessing function and recommended input size
        """
        preprocessing_map = {
            'efficientnet': {
                'preprocess_fn': tf.keras.applications.efficientnet.preprocess_input,
                'input_size': (224, 224),
                'rescale': False  # EfficientNet preprocessing already includes scaling
            },
            'mobilenet': {
                'preprocess_fn': tf.keras.applications.mobilenet_v2.preprocess_input,
                'input_size': (224, 224),
                'rescale': False
            },
            'resnet': {
                'preprocess_fn': tf.keras.applications.resnet50.preprocess_input,
                'input_size': (224, 224),
                'rescale': False
            },
            'xception': {
                'preprocess_fn': tf.keras.applications.xception.preprocess_input,
                'input_size': (299, 299),
                'rescale': False
            }
        }
        
        return preprocessing_map.get(model_name.lower(), {
            'preprocess_fn': lambda x: x / 255.0,
            'input_size': (224, 224),
            'rescale': True
        })
    
    def load_and_prepare_data(self, model_name='efficientnet', validation_split=0.15, 
                             augment_training=True):
        """
        Complete data loading and preparation pipeline
        
        Args:
            model_name: Target model name
            validation_split: Validation set ratio
            augment_training: Whether to apply data augmentation to training set
            
        Returns:
            train_dataset, val_dataset, class_names, dataset_info
        """
        # Get model-specific preprocessing settings
        model_config = self.get_model_specific_preprocessing(model_name)
        
        # Update image size
        self.img_size = model_config['input_size']
        
        print(f"Preparing data for {model_name} model...")
        print(f"Target image size: {self.img_size}")
        
        # Create dataset
        train_ds, val_ds, class_names = self.create_dataset_from_directory(validation_split)
        
        # Apply model-specific preprocessing
        if not model_config['rescale']:
            # Use model-specific preprocessing function
            preprocess_fn = model_config['preprocess_fn']
            
            def apply_preprocessing(image, label):
                # Simply apply preprocessing - let TensorFlow handle shapes
                image = preprocess_fn(image)
                return image, label
            
            train_ds = train_ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            # Apply data augmentation and normalization
            train_ds = self.apply_data_augmentation(train_ds, augment_training)
            val_ds = self.apply_data_augmentation(val_ds, False)
        
        # Optimize performance for Raspberry Pi
        train_ds = self.optimize_dataset_for_raspberry_pi(train_ds)
        val_ds = self.optimize_dataset_for_raspberry_pi(val_ds)
        
        # Collect dataset information
        dataset_info = {
            'num_classes': len(class_names),
            'class_names': class_names,
            'input_shape': (*self.img_size, 3),
            'model_name': model_name,
            'batch_size': self.batch_size
        }
        
        print("Dataset preparation completed!")
        print(f"Number of classes: {dataset_info['num_classes']}")
        print(f"Class names: {class_names}")
        
        return train_ds, val_ds, class_names, dataset_info

def visualize_dataset(dataset, class_names, num_images=9, save_path=None):
    """
    Visualize dataset samples
    
    Args:
        dataset: TensorFlow dataset
        class_names: List of class names
        num_images: Number of images to display
        save_path: Path to save the visualization image (optional)
    """
    import datetime
    
    plt.figure(figsize=(10, 10))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    plt.tight_layout()
    
    # Generate unique filename if save_path is not provided
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"dataset_visualization_{timestamp}.png"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Dataset visualization saved to: {save_path}")

# Usage example
if __name__ == "__main__":
    # Initialize data loader
    data_dir = "./cat_dataset"  # Your dataset path
    loader = CatDatasetLoader(data_dir, batch_size=16)  # Smaller batch_size recommended for Raspberry Pi
    
    # Prepare data for EfficientNet
    train_ds, val_ds, class_names, info = loader.load_and_prepare_data(
        model_name='efficientnet',
        validation_split=0.15,
        augment_training=True
    )
    
    print("Dataset info:", info)
    
    # Visualize some samples and save to file
    visualize_dataset(train_ds, class_names, save_path="train_dataset_samples.png")
    
    # Check dataset size
    train_size = tf.data.experimental.cardinality(train_ds).numpy()
    val_size = tf.data.experimental.cardinality(val_ds).numpy()
    
    print(f"Training batches: {train_size}")
    print(f"Validation batches: {val_size}")
