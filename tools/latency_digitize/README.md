# latency\_digitize

This folder contains a small utility script to extract the "latency vs. number of people" plot data from `StereoMulti3DPose/figures/latency.pdf` and regenerate a simplified figure for this thesis.

- Input: `StereoMulti3DPose/figures/latency.pdf`
- Output:
  - `tools/latency_digitize/latency_vs_people.csv`
  - `images/latency_vs_people.pdf`

Notes:
- The source PDF is a grouped bar chart exported as vector graphics. The script converts it to SVG via `pdftocairo -svg`, then parses the bar rectangles.
- Method-to-color mapping follows the legend order as extracted from the source figure.

