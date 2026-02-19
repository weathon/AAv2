#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find "$SCRIPT_DIR" -path '*/HPSv3' -prune -o -type f \( -name '*.md' -o -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.cfg' -o -name '*.ini' -o -name '*.conf' -o -name '*.rst' -o -name '*.csv' -o -name '*.xml' -o -name '*.html' -o -name '*.css' -o -name '*.js' -o -name '*.ts' -o -name '*.env' -o -name '*.gitignore' -o -name '*.dockerignore' -o -name 'Makefile' -o -name 'Dockerfile' -o -name '*.lock' -o -name '*.log' \) ! -name '*.txt' ! -name '*.json' ! -name '*.jsonl' ! -name 'export.sh' ! -name 'takeout.txt' -print | sort | while read f; do echo "===== $f ====="; cat "$f"; echo; done > "$SCRIPT_DIR/takeout.txt"

echo "Exported to $SCRIPT_DIR/takeout.txt"
