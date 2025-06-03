from string import punctuation
print(list(punctuation))
def adjust_punctuation(text):
    # Splitting the text into lines
    lines = text.strip().split('\n')

    # Process each line
    processed_lines = []
    for line in lines:
        elements = line.split('\t')
        if len(elements) < 2:
            # Directly append lines without enough elements (like metadata)
            processed_lines.append(line)
            continue

        token, *labels = elements
        if token in list(punctuation) and processed_lines:
            # Concatenate the punctuation with the previous token, keeping labels unchanged
            previous_line = processed_lines[-1].split('\t')
            # print(previous_line)
            processed_lines[-1] = previous_line[0] + token + '\t' + '\t'.join(previous_line[1:])
        else:
            # Add the current token and its labels as a new entry
            processed_line = '\t'.join([token] + labels)
            processed_lines.append(processed_line)

    return '\n'.join(processed_lines)

for file in ['../data/release-december-2023/HIPE-2022-v2.1-letemps-dev-fr.tsv',
             '../data/release-december-2023/HIPE-2022-v2.1-hipe2020-test-fr.tsv',
             '../data/release-december-2023/HIPE-2022-v2.1-hipe2020-train.tsv']:
    with open(file, 'r') as f:
        text = f.read()
    text = adjust_punctuation(text)
    text = adjust_punctuation(text)
    file = file.replace('.tsv', '_punct.tsv')
    with open(file, 'w') as f:
        f.write(text)