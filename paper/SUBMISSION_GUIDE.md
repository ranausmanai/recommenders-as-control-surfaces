# arXiv Submission Guide

## Files in this directory

```
paper/
├── paper.tex              # main LaTeX source
├── references.bib         # bibliography (minimal starter)
├── figures/
│   ├── paper_fig1_cross_model_attack.png
│   ├── paper_fig2_dose_response.png
│   ├── paper_fig3_generator_swap.png
│   └── paper_fig4_defenses.png
└── SUBMISSION_GUIDE.md    # this file
```

## Build locally (sanity check)

You need a TeX distribution (TeX Live, MacTeX, MikTeX).

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex   # twice more for cross-refs and figure numbering
```

This produces `paper.pdf`. Open it and verify:
- The 4 figures render at the right places.
- All citations show as `[1]`, `[2]`, ... not `[?]`.
- Tables 1–3 are correctly formatted.
- The italic quote appears at the end of the abstract and the conclusion.

If `pdflatex` is not installed:
- macOS: `brew install --cask mactex-no-gui`
- Ubuntu: `sudo apt install texlive-full`
- Or use Overleaf: upload all files in `paper/` (including `figures/`) and compile in-browser.

## Pre-submission checklist

Before uploading to arXiv:

1. **Fill in author info on the title page.** The current `\author{}` is set to
   `Usman Ashraf Rana` with `usmanashrafrana@gmail.com`. Change affiliation,
   ORCID, and acknowledgements if needed.
2. **Add a corresponding author footnote** if you want a different contact email.
3. **Verify references.** The included `.bib` is a minimal starter. Add real
   citations as needed before formal submission. arXiv will accept the paper
   either way, but a richer bibliography helps reviewers.
4. **Pick an arXiv category.** This paper fits best at:
   - **cs.CR** (cryptography and security) — primary, because of the
     adversarial-attack framing.
   - **cs.CL** (computation and language) — secondary, because the agents
     are LLM-based.
   - **cs.LG** (machine learning) — tertiary, for ML methodology.
   Submit primary `cs.CR`, cross-list `cs.CL` and `cs.LG`.
5. **License.** arXiv non-exclusive distribution license is fine; or pick
   CC-BY 4.0 if you want maximum reuse.
6. **Comments field** (when uploading): something like
   *"Preprint. Code and data available at <repo URL>. Comments welcome."*
7. **Make sure the figures directory is included in the upload.** arXiv
   compiles your sources server-side and needs every `\includegraphics{}`
   asset to be present.

## How to upload to arXiv

1. Go to <https://arxiv.org/submit>.
2. Sign in (or create an account; new accounts need an endorser for the
   first submission to most categories — `cs.CR` and `cs.CL` are
   non-endorsement currently, but verify at submission time).
3. Choose **Start new submission**.
4. **License:** select CC-BY 4.0 (recommended) or arXiv non-exclusive.
5. **Author:** confirm your name and ORCID.
6. **Title / abstract:** paste from the corresponding fields in
   `paper.tex`. arXiv strips LaTeX from the abstract; plain text is fine.
7. **Categories:** primary `cs.CR`, optional cross-list `cs.CL` and `cs.LG`.
8. **Files:** upload the whole `paper/` directory as a `.tar.gz` (preferred)
   or upload each file individually:
   - `paper.tex`
   - `references.bib`
   - `figures/*.png`
9. arXiv will compile your LaTeX server-side. If it fails, you'll see the
   log and can iterate.
10. **Submit and wait.** Standard turnaround is the next business day
    (or weekend submissions appear Monday).

## Suggested companion repo

Release the code + data on GitHub at submission time. Recommended structure:

```
.
├── paper.pdf           # final PDF
├── src/                # all of /src from this project
├── notebooks/          # all of /notebooks
├── posts/              # the 5 post pools
├── results/
│   ├── decision_shift*.jsonl   # raw rollouts
│   ├── 10_rigorous_stats.json
│   ├── 10_decision_tables.md
│   └── figures/*.png
└── README.md
```

A README that points to the headline reproduction (`notebooks/11_paper_figures.py`
+ the JSONLs) lets reviewers regenerate every number and figure in <2 minutes.

## Optional polish before submission

- Add a short related-work section citing recent prompt-injection /
  agent-attack literature (Greshake et al., AgentDojo, PoisonedRAG).
- Tighten the abstract to ~250 words (it's currently ~340).
- Add ORCID to the author block.
- Add an acknowledgements line at the end of the Conclusion.

These are quality-of-life improvements; the paper is submittable as is.
