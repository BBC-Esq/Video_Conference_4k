import os

def clean_whitespace_only_lines(filepath):
    """
    Remove whitespace from lines that contain only whitespace,
    while preserving the blank line itself.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        with open(filepath, 'r', encoding='latin-1') as file:
            lines = file.readlines()

    cleaned_lines = []
    changes_made = 0

    for line in lines:
        # Check if line contains only whitespace (but isn't empty)
        if line.strip() == '' and line != '\n' and line != '':
            cleaned_lines.append('\n')
            changes_made += 1
        else:
            cleaned_lines.append(line)

    # Only write back if changes were made
    if changes_made > 0:
        with open(filepath, 'w', encoding='utf-8', newline='') as file:
            file.writelines(cleaned_lines)
        return changes_made

    return 0

def process_directory(directory):
    """
    Recursively process all .py files in the directory.
    """
    total_files = 0
    total_changes = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                changes = clean_whitespace_only_lines(filepath)
                total_files += 1
                total_changes += changes

                if changes > 0:
                    print(f"Cleaned {changes} line(s) in: {filepath}")

    return total_files, total_changes

def main():
    # Get the current working directory
    current_dir = os.getcwd()

    print(f"Scanning for .py files in: {current_dir}")
    print("-" * 50)

    files_processed, total_changes = process_directory(current_dir)

    print("-" * 50)
    print(f"Complete!")
    print(f"Files scanned: {files_processed}")
    print(f"Total whitespace-only lines cleaned: {total_changes}")

if __name__ == "__main__":
    main()