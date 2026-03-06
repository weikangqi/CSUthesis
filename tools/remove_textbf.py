import re

with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    content = f.read()

before = content.count(chr(0x5c) + 'textbf{')
print(f'Found {before} \\textbf occurrences')

# Remove \textbf{...} keeping inner content
new_content = re.sub(r'\\textbf\{([^}]*)\}', r'\1', content)

after = new_content.count(chr(0x5c) + 'textbf{')
print(f'Removed {before - after}, remaining {after}')

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Done')
