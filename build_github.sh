#!/bin/bash
# Build the GitHub Pages copy: docs/ = index.html wrapped in a full HTML document + assets/ + data/.
cd "$(dirname "$0")"; mkdir -p docs
{ printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'; cat index.html; printf '\n</body>\n</html>\n'; } > docs/index.html
cp -r assets data docs/; touch docs/.nojekyll
echo "docs/ ready ($(du -sh docs | cut -f1)). Push the repo and set Pages to serve /docs."
