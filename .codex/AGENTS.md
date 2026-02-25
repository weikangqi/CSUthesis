# Agent Instructions for Master Thesis Writing (Chinese Academic Style)

You are assisting in writing a Chinese Master’s thesis in LaTeX.

The thesis must follow the standards of Chinese graduate academic writing:
rigorous structure, sufficient theoretical development, complete logical flow,
clear mathematical formulation, and formal academic tone.

The writing should primarily consist of well-developed natural paragraphs.
Bullet-point lists should be minimized unless required for formal definitions.

The language must conform to formal Chinese academic writing conventions,
with precise terminology and logically connected arguments.

---------------------------------------------------------------------

## Reference to Previously Published English Paper

The author has a previously published English academic paper.

The LaTeX source code of that paper may be used as an academic reference.
Since the previous work is written in English, translation into Chinese is permitted.

The LaTeX source of the published paper is located at:
`StereoMulti3DPose/paper/main.tex`

The published paper project is stored under the folder:
`StereoMulti3DPose/`

Folder purposes:

- `StereoMulti3DPose/paper/`: LaTeX source of the published English paper (main entry `main.tex`), bibliography file (`refs.bib`), and the IEEE template class (`IEEEtran.cls`). This is the primary reference when reusing structure, formulations, and experimental descriptions.
- `StereoMulti3DPose/figures/`: Figures used by the paper (e.g., pipeline diagram, ablations, qualitative results). These files can be reused as references when recreating thesis figures (but captions and text should be rewritten in Chinese academic style).
- `StereoMulti3DPose/authors/`: Author photos and related assets used for the paper PDF.
- `StereoMulti3DPose/build/`: Compiled outputs of the paper (e.g., `main.pdf`) and intermediate build artifacts. These are not thesis sources and should generally not be modified; they are useful for quickly checking the final paper rendering.

The overall logical structure and section organization may be preserved where appropriate.
Structural alignment with the previous work is allowed.

However:

- Direct copying of English sentences is strictly forbidden.
- Mechanical sentence-by-sentence translation must be avoided.
- The linguistic expression must be reconstructed in formal Chinese academic style.
- Explanations should be expanded when necessary to meet thesis-level depth.

In a Master's thesis, theoretical derivations, background explanations,
and methodological reasoning should be more detailed than in a conference paper.

---------------------------------------------------------------------

## Reuse of Experimental Methods and Results

The following content may be reused when scientifically justified:

- Problem formulation
- Mathematical definitions
- Model architecture
- Experimental setup
- Dataset descriptions
- Evaluation metrics
- Empirical results

Reused experimental results must remain numerically accurate.
Do not fabricate new results.

When reusing experiments, the thesis should:

- Provide deeper explanation of methodology
- Include more detailed implementation description
- Add analytical discussion of results
- Expand theoretical interpretation

The thesis must demonstrate academic progression and expanded exposition
compared to the previously published paper.

---------------------------------------------------------------------

## Mathematical Rigor and Formalization

All technical sections must be mathematically grounded.

Before introducing equations:

- Clearly define symbols and variables.
- Explain assumptions.
- Maintain notation consistency.

Whenever possible, present formal definitions such as:

\[
\mathcal{L}(\theta) = \mathcal{L}_{sup}(\theta) + \lambda \mathcal{R}(\theta),
\]

Each equation must be followed by detailed explanation in Chinese academic prose.

Avoid vague statements such as “显著提升性能” without quantitative or theoretical support.

---------------------------------------------------------------------


## Academic Integrity

Although structural alignment and translation are permitted,
the thesis must not replicate the wording of the English paper.

The model must ensure:

- Independent Chinese academic expression.
- Improved logical clarity.
- Expanded theoretical exposition.
- Proper citation of the previously published paper when necessary.

---------------------------------------------------------------------

## Output Requirements

All outputs must be valid LaTeX code.

Do not produce Markdown headers.
Do not include meta commentary.
Return only the LaTeX content of the requested section.

Use proper LaTeX math environments for equations.
Ensure notation consistency across chapters.

The final writing must resemble a formal Chinese Master’s thesis,
not a conference paper draft.
