#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/web"
# Clean install so optional deps (e.g. rollup native bindings) resolve correctly
rm -rf node_modules
npm install
npm run build
npm run deploy
