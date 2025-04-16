import os

def print_limited_tree(start_path, max_files_per_folder=5, prefix=""):
    try:
        entries = sorted(os.listdir(start_path))
    except PermissionError:
        return

    dirs = [e for e in entries if os.path.isdir(os.path.join(start_path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(start_path, e))]

    entries_to_show = dirs + files[:max_files_per_folder]
    hidden_count = max(0, len(files) - max_files_per_folder)

    pointers = ['├── '] * (len(entries_to_show) - 1) + ['└── '] if entries_to_show else []

    for i, name in enumerate(entries_to_show):
        path = os.path.join(start_path, name)
        connector = pointers[i]
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = '│   ' if i != len(entries_to_show) - 1 else '    '
            print_limited_tree(path, max_files_per_folder, prefix + extension)

    if hidden_count > 0:
        print(prefix + f'└── ... ({hidden_count} more files)')

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 📁 ის დირექტორია, სადაც სკრიპტია
    ROOT_DIR = os.path.join(SCRIPT_DIR, "")    # 📁 მიზნობრივი ფოლდერი
    MAX_FILES = 5

    print("xtts2_training/")
    print_limited_tree(ROOT_DIR, max_files_per_folder=MAX_FILES)