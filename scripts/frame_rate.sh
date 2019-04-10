#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
pushd "${SCRIPT_DIR}/.." > /dev/null

for file in data/videos/*; do
    ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "${file}"
done

popd > /dev/null

