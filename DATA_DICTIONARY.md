# Data Dictionary

This file documents the data files referenced in the Data Sharing Statement.

## data/endoscapes_test_frames.txt

The 312 frame identifiers of the official Endoscapes-SG201 test split, one per line.

## data/endoscapes_readerstudy_100frames.txt

The 100 frame identifiers randomly sampled from the test split for the expert rating procedure, one per line.

## data/model_descriptions/

GPT-5 (gpt-5-2025-08-07) outputs per input setting (image only, image plus predicted graph, image plus ground-truth graph).

- `full_reasoning/`: full step-by-step model responses, including the structured fields (anatomy list, action triplets, CVS confidence vector, BDI risk level).
- `concise_312frames/`: the final concise statement per step for all 312 test frames (input to the automatic structured evaluation).
- `concise_100frames_readerstudy/`: the final concise statements for the 100 rated frames, exactly as shown to the raters.

## data/reader_study/expert_ratings.csv

One row per rater, frame, and reasoning step: 900 rows = 100 frames x 3 raters per frame x 3 steps. All Likert scores are on a 1 to 5 scale.

| Column | Description |
| --- | --- |
| `timestamp` | Rating submission time (export format of the rating interface). |
| `expertId` | Rater identifier, 1 to 10. All raters are board-certified surgeon coauthors. |
| `caseId` | Endoscapes frame identifier. |
| `taskKey` | Reasoning step: `Task1` observation, `Task2` insight, `Task3` future plan and outcome. |
| `bdi` | The rater's own bile duct injury risk level for the frame (1 to 5). Recorded once per rater-frame assignment, on the `Task2` row; blank elsewhere. The per-case reference in `expert_bdi_per_case.json` is the median of the 3 raters of each frame. |
| `annotation` | The rater's free-text annotation for the step (Korean). |
| `Annotation_En` | Machine translation of `annotation` into English (used for the text-metric analysis). |
| `img_score1`, `img_score2` | Scores for the image-only description at this step: `score1` = accuracy (anatomical accuracy at step 1, clinical accuracy at step 2, clinical validity at step 3), `score2` = detail (descriptive detail at step 1, reasoning detail at step 2, detail and feasibility at step 3). |
| `pred_score1`, `pred_score2` | Same scores for the image plus predicted graph description. |
| `gt_score1`, `gt_score2` | Same scores for the image plus ground-truth graph description. |
| `img_gc`, `pred_gc`, `gt_gc` | Global coherence of each description, rated once per rater-frame assignment, on the `Task3` row; blank elsewhere. |

Raters scored the 3 descriptions side by side, with the setting of each column randomized per assignment and hidden from the rater.

## data/reader_study/expert_bdi_per_case.json

Per-frame BDI reference for each of the 100 frames: the 3 raters' own `bdi` levels (`list`) with their `mean` and `median`. The median is the shared reference for the BDI mean absolute error of all 3 settings, so that error is interpreted only as a relative comparison across settings.

## data/reader_study/expert_ratings_original_export.csv

Unprocessed export of the rating interface, kept for provenance; `expert_ratings.csv` is the cleaned analysis input.

## Reproducing the main numbers

See `README.md` (Reproducing). The scripts in `scripts/analysis/` read these files with the repo-root relative paths above, for example:

```
python3 scripts/analysis/compute_permutation.py
```
