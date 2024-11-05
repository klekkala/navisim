#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Error: Please provide an argument."
  exit 1
fi

for ((i = 1; i<= 1; i++))
do 
  index=0
  ssh student@iGpu10 "ls -1v /data/$1/cam$i/*.bag" | while read entry; do
      if [ "${entry: -4}" != ".bag" ]; then
          continue
      fi
      python3 download_bag.py --bag "student@iGpu10:${entry}" --cam "$i" --save $1
      ((index++))
  done
done

./vision_toolkit/extract/bag_.sh "$1"
./vision_toolkit/scripts/run_divide.sh "$1"
./vision_toolkit/scripts/run_LeGO.sh "$1"
./vision_toolkit/scripts/run_LeGO_parameter.sh '/lab/tmpig23b/navisim/data/bag_dump/2023_03_11/bags/cam1/test_2023-03-11-11-05-59.bag' 0 "$1"
./vision_toolkit/scripts/all_pcl.sh "$1"
