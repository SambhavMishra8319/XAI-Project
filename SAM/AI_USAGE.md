# AI_USAGE.md — DS357 Project

## Tools Used
- Claude (Anthropic) — claude.ai

## Prompts Used
1. "Help me understand how SRA is computed in this paper [paper excerpt]"
2. "Generate skeleton code for LIME stability measurement using CV and SRA"
3. "Suggest a research gap and extension for this paper's Phase 3"

## How Output Was Modified
- All code was reviewed line by line against the paper's equations (Eq. 6-11)
- SRA formula corrected to match pooled variance formulation in paper
- VSI/CSI implementations re-derived from Section 5.3 independently
- Phase 3 hypothesis and strategy selection were our own research decision
- All variable names and comments rewritten for clarity

## Verification Steps
- Ran quick mode on synthetic data to verify metric directions
  (CV increases as noise increases ✓, VSI decreases with more random feature
  selection ✓, SRA increases with more rank disagreement ✓)
- Cross-checked against paper's Table 5 (dataset sizes) ✓
- Confirmed LIME uses Ridge regression as surrogate (lime package default) ✓
- Confirmed SHAP uses interventional Tree SHAP (shap.TreeExplainer default) ✓
