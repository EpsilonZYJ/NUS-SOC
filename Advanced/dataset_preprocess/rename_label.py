import os

train_label_path = '/train/labels'
valid_label_path = '/valid/labels'
test_label_path = '/test/labels'

start_index = None

def rename_label(dataset_name):
    try:
        train_label_path = os.path.join(dataset_name, train_label_path)
        valid_label_path = os.path.join(dataset_name, valid_label_path)
        test_label_path = os.path.join(dataset_name, test_label_path)

        train_label_files = os.listdir(train_label_path)
        valid_label_files = os.listdir(valid_label_path)
        test_label_files = os.listdir(test_label_path)
    
        rename_label(train_label_path, train_label_files)
        rename_label(valid_label_path, valid_label_files)
        rename_label(test_label_path, test_label_files)
        print("Label files renamed successfully")
    except Exception as e:
        print(f"Error: {e}")

def rename_label(dataset_path, dataset_filepath_list):
    try:
        for file in dataset_filepath_list:
            file_path = os.path.join(dataset_path, file)
            if not file_path.endswith('.txt'):
                continue
            
            # read label file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                line = line.strip().split()
                
                # get old label
                old_label = line[0]

                new_label = str(start_index+int(old_label))
                new_line = [new_label] + line[1:]
                new_lines.append(' '.join(new_line) + '\n')
            
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
    except Exception as e:
        print(f"Error: {e}")

def main():
    start_index = int(input("Enter the start index: "))
    dataset_name = input("Enter the dataset name: ")

if __name__ == '__main__':
    main()
