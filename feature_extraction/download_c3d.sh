#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
pushd "${SCRIPT_DIR}/.." > /dev/null

features_folder_path=data/features

mkdir "${features_folder_path}"
wget -O "${features_folder_path}/c3d.pickle" http://v7.eecs.umich.edu/LifeQA/c3d.pickle 

popd > /dev/null
