import os
import glob
from pathlib import Path


def rename_images(folder_path, prefix="image"):
    """
    Batch rename image files in the specified folder

    Parameters:
    folder_path (str): Path to the image folder
    prefix (str): Prefix for renaming, default is "image"
    """

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return

    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']

    # Get all image files
    image_files = []
    for extension in image_extensions:
        # Use glob to find files (case insensitive)
        pattern = os.path.join(folder_path, extension)
        image_files.extend(glob.glob(pattern))
        # Also search for uppercase extensions
        pattern_upper = os.path.join(folder_path, extension.upper())
        image_files.extend(glob.glob(pattern_upper))

    # Remove duplicates and sort
    image_files = list(set(image_files))
    image_files.sort()

    if not image_files:
        print(f"No image files found in folder '{folder_path}'!")
        return

    print(f"Found {len(image_files)} image files")
    print("Starting to rename...")

    # Track renaming operations
    renamed_count = 0
    failed_count = 0

    for i, old_path in enumerate(image_files, 1):
        try:
            # Get file extension
            file_extension = Path(old_path).suffix.lower()

            # Generate new filename
            new_filename = f"{prefix}_{i}{file_extension}"
            new_path = os.path.join(folder_path, new_filename)

            # Check if new filename already exists
            if os.path.exists(new_path) and old_path != new_path:
                print(
                    f"Warning: File '{new_filename}' already exists, skipping rename of '{os.path.basename(old_path)}'")
                failed_count += 1
                continue

            # Rename file
            if old_path != new_path:  # Avoid renaming the same file
                os.rename(old_path, new_path)
                print(f"✓ {os.path.basename(old_path)} -> {new_filename}")
                renamed_count += 1
            else:
                print(f"- {new_filename} (filename already in target format)")

        except Exception as e:
            print(f"✗ Rename failed: {os.path.basename(old_path)} - Error: {str(e)}")
            failed_count += 1

    # Output statistics
    print(f"\nRenaming completed!")
    print(f"Successfully renamed: {renamed_count} files")
    if failed_count > 0:
        print(f"Failed/Skipped: {failed_count} files")


def rename_main():
    """Main function - Interactive usage"""
    print("=== Batch Image Renaming Tool ===")

    # Get user input
    while True:
        folder_path = input("Please enter the image folder path: ").strip()
        if folder_path:
            # Handle quotes
            folder_path = folder_path.strip('"\'')
            break
        print("Path cannot be empty, please enter again!")

    # Optional: Custom prefix
    custom_prefix = input("Please enter filename prefix (press Enter for default 'image'): ").strip()
    prefix = custom_prefix if custom_prefix else "image"

    # Confirm operation
    print(f"\nAbout to rename image files in folder '{folder_path}'")
    print(f"New filename format: {prefix}_1.jpg, {prefix}_2.png, ...")
    confirm = input("Confirm execution? (y/n): ").strip().lower()

    if confirm in ['y', 'yes']:
        rename_images(folder_path, prefix)
    else:
        print("Operation cancelled")


# Example for direct function call
def rename_images_direct(folder_path, prefix="image"):
    """
    Example for direct function call
    Usage: rename_images_direct("C:/your/image/folder")
    """
    rename_images(folder_path, prefix)


def main():
    """
    Main function for this script.
    """
    rename_main()

if __name__ == "__main__":
    # Interactive execution
    main()

    # Or direct call example (uncomment the line below)
    # rename_images_direct("./images")  # Replace with your image folder path
