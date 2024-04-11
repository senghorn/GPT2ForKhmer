import os

def check_encoding(base_dir):
    non_utf8_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError:
                    non_utf8_files.append(file_path)
    
    return non_utf8_files

base_dir = 'gpt2/data/kmwiki_data'
non_utf8_files = check_encoding(base_dir)

if non_utf8_files:
    print("Non-UTF-8 Encoded Files:")
    for file in non_utf8_files:
        print(file)
else:
    print("All files are UTF-8 encoded.")
