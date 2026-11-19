import os

dataset_path = "dataset"

print("Dataset folder exists:", os.path.exists(dataset_path))

if os.path.exists(dataset_path):
    print("\nFolders inside dataset:")

    for item in os.listdir(dataset_path):
        print(" -", item)

test_path = "dataset/Test"

print("\nTest folder exists:", os.path.exists(test_path))

if os.path.exists(test_path):
    classes = os.listdir(test_path)

    print("Number of folders/classes:", len(classes))

    print("\nFirst 10 folders:")

    for class_name in classes[:10]:
        print(" -", class_name)