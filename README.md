# GravHMR project page

- `index.html`    page body (edit this). `build_github.sh` wraps it into `docs/index.html` and copies `assets/`, `data/`.
- `assets/`       teaser (from sports_wild/fig1 (1).pdf, 200 dpi) and pipeline figure.
- `data/`         cached clip exports: `<id>.json` header, `<id>.bin` skinned SMPL animation, `<id>.mp4` input (360p), `<id>.jpg` thumbnail.
- `tools/export.py`     exporter (wild: GravHMR paper config + ViPE camera, GeoCalib up; emdb: GT / GVHMR+PVA-Net rollout / GravHMR ViPE, Sim3 on first 100 frames).
- `tools/export_all.sh` exports every clip on the page, skipping ones already in data/ (delete a clip's files to redo it).
- `tools/pick_emdb.py`  ranks EMDB-2 sequences by root-error gain (how the three EMDB examples were chosen).

Local preview (fetch() needs http, not file://):  `cd docs && python -m http.server 8000`
GitHub Pages: push the repo, Settings -> Pages -> branch main, folder /docs.
