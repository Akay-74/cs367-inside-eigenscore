#!/usr/bin/env bash
# Render each slide of slides.html to png/slide_NN.png at 1920x1080.
cd "$(dirname "$0")"; mkdir -p png
for i in $(seq 1 10); do
  google-chrome --headless=new --disable-gpu --hide-scrollbars --window-size=1920,1080 \
    --virtual-time-budget=8000 --screenshot="png/slide_$(printf %02d $i).png" \
    "file://$PWD/slides.html?s=$i" >/dev/null 2>&1
done
ls png
