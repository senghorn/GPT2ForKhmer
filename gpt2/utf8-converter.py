import chardet

dataset_path = r'C:\Users\Sun_r\Projects\NLPProject\khmer-text-data\gpt2\data-set.txt'

# Detect the encoding of the file
with open(dataset_path, 'rb') as file:
    raw_data = file.read()
    encoding = chardet.detect(raw_data)['encoding']

# If the detected encoding is not UTF-8, convert it
if encoding != 'utf-8':
    with open(dataset_path, 'r', encoding=encoding) as file:
        content = file.read()

    # Save the file in UTF-8 encoding
    with open(dataset_path, 'w', encoding='utf-8') as file:
        file.write(content)
