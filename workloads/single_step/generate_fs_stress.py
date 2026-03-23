import os
import random
import string
import shutil

ROOT_DIR = "workloads/single_step/mock_fs_payload"
MAX_DEPTH = 5
FOLDERS_PER_LEVEL = 3
FILES_PER_FOLDER = 15

def create_dummy_file(filepath):
    """Creates a small dummy file to populate the directory."""
    with open(filepath, 'w') as f:
        f.write(''.join(random.choices(string.ascii_letters, k=64)))

def generate_tree(current_path, current_depth):
    """Recursively generates a deeply nested directory tree."""
    if current_depth > MAX_DEPTH:
        return

    # Create files in the current directory
    for i in range(FILES_PER_FOLDER):
        create_dummy_file(os.path.join(current_path, f"mock_log_{i}.txt"))

    # Create subdirectories and recurse
    for i in range(FOLDERS_PER_LEVEL):
        sub_dir = os.path.join(current_path, f"dir_{i}")
        os.makedirs(sub_dir, exist_ok=True)
        generate_tree(sub_dir, current_depth + 1)

if __name__ == "__main__":
    print(f"Generating NFS-friendly I/O directory tree at: {ROOT_DIR}")
    
    if os.path.exists(ROOT_DIR):
        print("Clearing old payload...")
        shutil.rmtree(ROOT_DIR)
        
    os.makedirs(ROOT_DIR, exist_ok=True)
    
    print("Spawning files and folders. This might take a moment depending on network latency...")
    generate_tree(ROOT_DIR, 1)
    
    # Calculate totals
    total_dirs = sum([len(d) for r, d, f in os.walk(ROOT_DIR)])
    total_files = sum([len(f) for r, d, f in os.walk(ROOT_DIR)])
    
    print(f"Done! Created {total_dirs:,} directories and {total_files:,} files.")