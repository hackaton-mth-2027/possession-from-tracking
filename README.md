# Team possession from tracking

Team: **Pass Quality Decoders**

More than a Hack 2027, Initial Challenge: label every frame of FC Barcelona (H) vs Athletic Club
(A) with the team in possession, or D for dead ball. The labels come from models learned on other
matches' public data, applied to the provided tactical tracking and decoded into possessions. No
labels of this match are used for training.

- Submission: `outputs/submissions/dfl_bag_st_dflcd.csv`, written by `scripts/make_submission.py`
- How it works and how it was validated: [TECHNICAL.md](TECHNICAL.md)

On the official public window (5 minutes, 17 possessions) the submission scores spellF1 0.690
(macroF1 0.934), against 0.326 for the organisers' best baseline (`baseline_smooth50`).

## Reproduce

Python 3.14, CPU only.

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 repo   # the organisers' repository
# unpack the match data bundle into repo/data/ (see repo/README.md)
python -m venv .venv && .venv/bin/pip install -r requirements.txt
tools/download_dfl.sh                         # 2.4 GB: DFL open matches (training)
tools/download_pff.sh <drive-folder-id>       # 3.7 GB: PFF World Cup 2022 (dead-ball models)
tools/download_metrica.sh                     # 175 MB: Metrica sample matches (8 networks, validation)
.venv/bin/python scripts/make_submission.py   # -> outputs/submissions/dfl_bag_st_dflcd.csv
```

- The organisers' repository is cloned into `repo/`: this code uses its loaders, its scorer and
  its validator, and reads the match data from `repo/data/`.
- The PFF folder id comes with free access after signing up on PFF FC's dataset page (see
  `tools/download_pff.sh`).
- About four hours on a laptop. Each model is trained first if it is missing. Seeds are fixed
  throughout: a model retrained from scratch gives the same probabilities.
- The script validates the file with the organisers' `repo/evaluation/validate_submission.py`.
- Optional, not needed for the submission: `tools/download_skillcorner.sh` (1.9 GB) for the
  check on real tracking, `scripts/skillcorner_val.py`.

## What is not in this repository

- **The match data, and anything derived from it** (features, predictions), under the
  competition's terms of use.
- **Trained models.** They are rebuilt from the open data above by the same script.
- **The external datasets.** The download scripts fetch them from their original hosts.
- **The organisers' repository.** It is cloned next to this code, under its own licence.

## Layout

| path | contents |
|---|---|
| `possession/` | the library: features, external data loaders and noise, decoding |
| `scripts/` | training, validation and the submission (`make_submission.py`) |
| `tools/` | downloads, the paired comparison and the full-match sanity check |
| `TECHNICAL.md` | the technical document |

## Data sources and citations

The rows marked "experiments only" are ideas that were tried and dropped (TECHNICAL.md, section 3);
their scripts are not part of this repository.

| dataset | licence / access | used for |
|---|---|---|
| DFL / Sportec Solutions open data (IDSSE), 7 Bundesliga matches | CC BY 4.0 | **submission**: training of every model |
| PFF FC 2022 FIFA World Cup, 64 matches | free after sign-up | **submission**: dead-ball models, dead-ball boundary detector |
| Metrica Sports sample data, 3 anonymised matches | public; "please acknowledge the source" | **submission**: training of 8 of the 16 networks; the validation set |
| SoccerTrack v2, 10 matches (panoramic video) | CC BY 4.0 | experiments only (touch attribution, visible contacts) |
| SoccerNet Game State Reconstruction 2025, 57 broadcast clips | GPL-3.0 | experiments only (visible contacts on broadcast footage) |
| SkillCorner open data, 20 A-League 2024/25 matches (broadcast tracking, player possessions) | MIT | validation only: the submission's settings re-checked on real automatic tracking (`scripts/skillcorner_val.py`); never trained on |

1. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal
   and event data in elite soccer. *Scientific Data*, 12, 195.
   https://doi.org/10.1038/s41597-025-04505-y. Data from DFL Deutsche Fußball Liga / Sportec
   Solutions, CC BY 4.0; copy used: https://huggingface.co/datasets/pysport/idsse-data
2. PFF FC. 2022 FIFA World Cup dataset: broadcast tracking and event data for all 64 matches.
   https://blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
3. Metrica Sports. Sample tracking and event data. https://github.com/metrica-sports/sample-data
4. Scott, A., Uchida, I., Kuroda, K., Kim, Y., & Fujii, K. (2025). SoccerTrack v2: A Full-Pitch
   Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting.
   arXiv:2508.01802. Data: https://huggingface.co/datasets/atomscott/soccertrack-v2 (release
   v1.2, CC BY 4.0)
5. Somers, V., Joos, V., Cioppa, A., Giancola, S., Ghasemzadeh, S. A., Magera, F., Standaert, B.,
   Mansourian, A. M., Zhou, X., Kasaei, S., Ghanem, B., Alahi, A., Van Droogenbroeck, M., & De
   Vleeschouwer, C. (2024). SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and
   Identification on a Minimap. arXiv:2404.11335. Data:
   https://huggingface.co/datasets/SoccerNet/SN-GSR-2025 (GPL-3.0)
6. SkillCorner. Open data: broadcast tracking, dynamic events and phases of play for A-League
   2024/25 matches (with PySport). https://github.com/SkillCorner/opendata (MIT)

Pre-trained models used in experiments only, none of them in the submission: Ultralytics YOLO11
(ball detection, pose), Roboflow's football ball detector (github.com/roboflow/sports), OpenAI
Whisper large-v3-turbo (commentary transcription, run locally) and Meta's V-JEPA 2 video encoder.
