#!/usr/bin/env bash

source activate bde
python demo.py --cam_id=1 --fullscreen $*
