#!/bin/bash
# Rebuild docs/ (the GitHub Pages copy) and push. Usage: bash tools/publish.sh "message"
set -e; cd "$(dirname "$0")/.."
bash build_github.sh
git add data/ docs/ index.html
git commit -m "${1:-update clips}" || { echo "nothing to commit"; exit 0; }
git pull --rebase
git push
echo "pushed; the page updates in 1-2 minutes: https://jhl25-pixel.github.io/gravhmr/"
