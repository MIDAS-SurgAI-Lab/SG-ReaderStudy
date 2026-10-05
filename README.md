# SG-ReaderStudy

Data and code for **"Expert evaluation of scene-graph grounding for AI-generated surgical scene descriptions in laparoscopic cholecystectomy"** (npj Digital Medicine).

GPT-5 generated three-step surgical scene descriptions (Observation, Insight, Future Plan & Outcome) from single laparoscopic cholecystectomy frames under three input settings (image only, image + predicted scene graph, image + ground-truth scene graph), and ten board-certified surgeons rated them in a blinded, side-by-side reader study.

## Dataset dependency (Endoscapes + Endoscapes-SG201)

The **frames (images) come from Endoscapes** (Endoscapes2023), the laparoscopic cholecystectomy video dataset released by CAMMA (Mascagni, Murali et al.) under CC BY-NC-SA 4.0. The **object-and-relation (scene-graph) annotations come from Endoscapes-SG201** (Murali et al.), which is built on top of Endoscapes. This study uses the official test split (40 videos, 312 frames).

We do **not** redistribute the Endoscapes frames or any Endoscapes/SG201 ground-truth annotations; they remain under their original license. Obtain them from the official Endoscapes source and cite both works. The 312 test-frame identifiers we used are listed in `data/endoscapes_test_frames.txt`, and the 100 reader-study frames in `data/endoscapes_readerstudy_100frames.txt`.

## Repository layout

```
data/
  endoscapes_test_frames.txt            # 312 test-frame filenames (identifiers only, no images)
  endoscapes_readerstudy_100frames.txt  # the 100 reader-study frames (subset of the 312)
  reader_study/                     # de-identified expert ratings (raterId 1-10), BDI per case
  model_descriptions/               # GPT-5 outputs: full reasoning + concise statements (3 settings)
scripts/
  generation/                       # prompting + GPT-5 description generation
  evaluation/                       # anatomy F1, action-triplet F1, CVS AP, BDI MAE
  analysis/                         # reader-study stats, mixed models, propagation, text metrics, figures
```

> **Not included** (owned by Endoscapes-SG201): the frames, and the ground-truth annotations
> (COCO, GT scene graphs, CVS labels). The model's own structured outputs (anatomy / action-triplet /
> CVS fields) were produced alongside those GT annotations in a merged file; to release them, strip the
> ground-truth fields first. Reference Endoscapes-SG201 for all of the above.

## Reproducing

1. Obtain Endoscapes-SG201 and place the frames/annotations locally (paths configured in `scripts/`).
2. Set API credentials via environment variables (never hard-code):
   ```bash
   export OPENAI_API_KEY=...
   export GEMINI_API_KEY=...
   ```
3. Python 3.10; install dependencies (`pip install -r requirements.txt`). Analysis uses SciPy and statsmodels.
4. Run `scripts/generation` → `scripts/evaluation` → `scripts/analysis` → `scripts/figures`.

## Privacy and ethics

No patient-identifiable data is included. The reader study collected blinded expert evaluations of AI-generated text; ratings are released with rater identifiers de-identified as 1 to 10.

## Citation

```
[BibTeX to be added on publication]
```

## License

Code: MIT (see `LICENSE`). Released data are the authors' own derived outputs; Endoscapes-SG201 assets remain under their original license.
