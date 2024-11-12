#!/bin/bash


for entry in "/lab/tmpig23b/navisim/data/bag_dump/"*; do
    date=$(basename "$entry")
    ./sync_cam.sh "$date"
  done


