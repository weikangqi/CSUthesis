---
name: latex-build-folder
description: Build LaTeX projects with outputs directed to a `build` folder. Use when compiling LaTeX (latexmk/xelatex) or when a user requests PDFs or auxiliary files be generated in `build/`.
---

# LaTeX Build Folder

## Overview
Build LaTeX in this repo with all outputs placed in `build/` (PDF and aux files). Prefer latexmk with explicit output directory.

## Workflow
1. From repo root, ensure `build/` exists.
2. Compile using latexmk with output directory set to `build/`.
3. If a Makefile exists but does not support `build/`, do not edit it unless asked; run latexmk directly.

## Commands (PowerShell)
```powershell
if (-not (Test-Path -Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
latexmk -xelatex -gg -silent -f -outdir=build -auxdir=build csuthesis_main
```
If the main file differs, replace `csuthesis_main` with the correct root tex name (without extension).

## Notes
- Expected output PDF path: `build\csuthesis_main.pdf`.
- Use `latexmk -c -outdir=build -auxdir=build csuthesis_main` to clean build artifacts when requested.
