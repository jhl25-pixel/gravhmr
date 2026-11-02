# Project-page build notes

- `../index.html`   page body (edit this). `../build_github.sh` wraps it into `docs/index.html` (adds head/SEO tags) and copies `assets/`, `data/`.
- `../data/`        cached clip exports: `<id>.json` header, `<id>.bin` skinned SMPL animation, `<id>.mp4` input (360p), `<id>.jpg` thumbnail.
- `export.py`       exporter (wild: GravHMR paper config + ViPE camera, GeoCalib up; emdb: GT / GVHMR+PVA-Net rollout / GravHMR ViPE, Sim3 on first 100 frames).
- `export_all.sh`   exports every clip on the page, skipping ones already in data/ (delete a clip's files to redo it).
- `pick_emdb.py`, `choose_emdb.py`  how the EMDB examples were chosen (root-error gain; share of frames where GravHMR is worse).
- `hero.py`         renders the hero loop (`--worker k --nworker n` to parallelise, then `--encode_only`). Frames cached in assets/hero/frames (git-ignored).
- `serve.py`        local static server with HTTP Range (video seeking): `python tools/serve.py 8765`.
