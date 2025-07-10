import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse

class YOLODatasetMerger:
    def __init__(self, output_dir: str):
        """
        Initialize YOLO dataset merger
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory structure
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)
        
        # Store all class information
        self.all_classes: Dict[str, int] = {}  # class_name -> new_id
        self.dataset_info: List[Dict] = []  # Store information for each dataset
        
    def load_dataset_config(self, dataset_path: str) -> Dict:
        """
        Load dataset configuration file
        
        Args:
            dataset_path: Dataset path
            
        Returns:
            Dataset configuration dictionary
        """
        config_path = Path(dataset_path) / "data.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def register_classes(self, dataset_name: str, classes: List[str], class_ids: Dict[str, int]):
        """
        Register classes, handle duplicate classes
        
        Args:
            dataset_name: Dataset name
            classes: Class name list
            class_ids: Original class ID mapping
        """
        dataset_info = {
            'name': dataset_name,
            'original_classes': classes,
            'original_ids': class_ids,
            'new_ids': {}
        }
        
        for class_name in classes:
            if class_name not in self.all_classes:
                # New class, assign new ID
                new_id = len(self.all_classes)
                self.all_classes[class_name] = new_id
                dataset_info['new_ids'][class_name] = new_id
            else:
                # Duplicate class, use existing ID
                dataset_info['new_ids'][class_name] = self.all_classes[class_name]
                
        self.dataset_info.append(dataset_info)
        
    def convert_label_file(self, label_path: Path, new_ids: Dict[str, int], 
                          original_classes: List[str]) -> List[str]:
        """
        Convert class IDs in label file
        
        Args:
            label_path: Label file path
            new_ids: New class ID mapping
            original_classes: Original class list
            
        Returns:
            List of converted label lines
        """
        if not label_path.exists():
            return []
            
        converted_lines = []
        
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # If file is empty, return empty list (will create empty file)
        if not content:
            return []
            
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                continue
                
            old_class_id = int(parts[0])
            if old_class_id >= len(original_classes):
                print(f"Warning: Class ID {old_class_id} out of range, skipping")
                continue
                
            class_name = original_classes[old_class_id]
            new_class_id = new_ids[class_name]
            
            # Update class ID, keep other coordinates unchanged
            new_line = f"{new_class_id} {' '.join(parts[1:])}"
            converted_lines.append(new_line)
                
        return converted_lines
    
    def copy_and_convert_dataset(self, dataset_path: str, dataset_info: Dict):
        """
        Copy and convert dataset files
        
        Args:
            dataset_path: Dataset path
            dataset_info: Dataset information
        """
        dataset_path = Path(dataset_path)
        new_ids = dataset_info['new_ids']
        original_classes = dataset_info['original_classes']
        
        # Process training set
        self._process_split(dataset_path, "train", new_ids, original_classes)
        
        # Process validation set
        self._process_split(dataset_path, "valid", new_ids, original_classes)
        
        # Process test set
        self._process_split(dataset_path, "test", new_ids, original_classes)
    
    def _process_split(self, dataset_path: Path, split: str, new_ids: Dict[str, int], 
                      original_classes: List[str]):
        """
        Process a dataset split (train/valid/test)
        
        Args:
            dataset_path: Dataset path
            split: Split name
            new_ids: New class ID mapping
            original_classes: Original class list
        """
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: {split} split does not exist in {dataset_path}")
            return
            
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + \
                     list(images_dir.glob("*.png"))
        
        for img_file in image_files:
            # Corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # Copy image file
            dest_img = self.output_dir / split / "images" / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Convert and copy label file
            converted_lines = self.convert_label_file(label_file, new_ids, original_classes)
            
            # Always create label file, even if empty
            dest_label = self.output_dir / split / "labels" / f"{img_file.stem}.txt"
            with open(dest_label, 'w', encoding='utf-8') as f:
                if converted_lines:
                    f.write('\n'.join(converted_lines))
                # If converted_lines is empty, file will be created as empty
                # This handles both empty original files and files with no valid annotations
    
    def merge_datasets(self, dataset_paths: List[str]):
        """
        Merge multiple datasets
        
        Args:
            dataset_paths: List of dataset paths
        """
        print("Starting dataset merge...")
        
        # Step 1: Load all dataset configurations and register classes
        for dataset_path in dataset_paths:
            print(f"Processing dataset: {dataset_path}")
            
            config = self.load_dataset_config(dataset_path)
            classes = config.get('names', [])
            nc = config.get('nc', 0)
            
            if len(classes) != nc:
                print(f"Warning: Class count mismatch - config: {nc}, actual classes: {len(classes)}")
            
            # Create original class ID mapping
            class_ids = {class_name: i for i, class_name in enumerate(classes)}
            
            # Register classes
            self.register_classes(Path(dataset_path).name, classes, class_ids)
        
        # Step 2: Copy and convert all datasets
        for i, dataset_path in enumerate(dataset_paths):
            print(f"Copying dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
            self.copy_and_convert_dataset(dataset_path, self.dataset_info[i])
        
        # Step 3: Generate merged configuration file
        self.generate_merged_config()
        
        # Step 4: Generate merge report
        self.generate_merge_report()
        
        print(f"Dataset merge completed! Output directory: {self.output_dir}")
    
    def generate_merged_config(self):
        """Generate merged configuration file with original format"""
        # Create config with the same format as original data.yaml files
        config_content = f"""train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: {len(self.all_classes)}
names: {list(self.all_classes.keys())}
"""
        
        config_path = self.output_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"Configuration file generated: {config_path}")
    
    def generate_merge_report(self):
        """Generate merge report"""
        report_path = self.output_dir / "merge_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO Dataset Merge Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total classes: {len(self.all_classes)}\n")
            f.write("Class mapping:\n")
            for class_name, class_id in sorted(self.all_classes.items(), key=lambda x: x[1]):
                f.write(f"  {class_id}: {class_name}\n")
            
            f.write("\nDataset information:\n")
            for dataset_info in self.dataset_info:
                f.write(f"\nDataset: {dataset_info['name']}\n")
                f.write("Original classes:\n")
                for i, class_name in enumerate(dataset_info['original_classes']):
                    new_id = dataset_info['new_ids'][class_name]
                    f.write(f"  {i} -> {new_id}: {class_name}\n")
        
        print(f"Merge report generated: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge YOLOv7 datasets")
    parser.add_argument("--datasets", nargs="+", required=True, 
                       help="List of dataset paths to merge")
    parser.add_argument("--output", required=True, 
                       help="Output directory path")
    
    args = parser.parse_args()
    
    # Create merger
    merger = YOLODatasetMerger(args.output)
    
    # Execute merge
    merger.merge_datasets(args.datasets)

if __name__ == "__main__":
    main()