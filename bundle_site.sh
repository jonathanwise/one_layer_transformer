#!/bin/sh
# Assemble the static viewer site into site/grokking/ (copy that folder to the website's public/grokking/).
set -e
cd "$(dirname "$0")"
OUT=site/grokking
rm -rf "$OUT"; mkdir -p "$OUT"
cp index.html neuron_viewer.html logit_viewer.html c3_viewer.html embed_viewer.html attention_viewer.html \
   mlp_viewer.html activation_dft_viewer.html timeline.js model_js.js model_worker.js runs.json "$OUT"/
cp -r snapshots_final snapshots "$OUT"/
cat > "$OUT/README.md" <<'EOT'
# Grokking mod-43 transformer viewers (static site)

Self-contained: open `index.html`. Everything is relative paths (HTML pages in iframes, a web worker, `runs.json`,
and float16 weight snapshots under `snapshots/` and `snapshots_final/` fetched over HTTP), so it must be served from
a web server (not file://), with all files kept together in this one directory. No build step, no external
dependencies. Link to it with an explicit path: `/grokking/index.html`.
Regenerate from the model repo with `bundle_site.sh`.
EOT
du -sh "$OUT"; find "$OUT" -type f | wc -l
