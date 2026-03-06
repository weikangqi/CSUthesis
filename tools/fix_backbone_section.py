with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines (1-indexed):
# 337: \subsubsection{常用网络骨干与机制}
# 338: intro sentence
# 339: \begin{itemize}
# 340:   \item ResNet
# 341:   \item Hourglass
# 342:   \item HRNet
# 343: \end{itemize}
# 344: attention mechanism paragraph (KEEP)
# 345: blank

# Verify target lines
print("Line 337:", repr(lines[336][:60]))
print("Line 338:", repr(lines[337][:60]))
print("Line 339:", repr(lines[338][:40]))
print("Line 343:", repr(lines[342][:40]))
print("Line 344:", repr(lines[343][:60]))
print("Line 345:", repr(lines[344]))

# Delete lines 337-343 (indices 336-342), keep line 344 onward
del lines[336:343]

# Verify result around that area
print("\nAfter edit:")
print("Line 337:", repr(lines[336][:80]))
print("Line 338:", repr(lines[337][:80]))

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('\nDone. Total lines:', len(lines))
