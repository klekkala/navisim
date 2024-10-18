source ~/catkin_ws/devel/setup.bash


if [ $# -eq 0 ]; then
  echo "Error: Please provide an argument."
  exit 1
fi

for ((i = 1; i<= 5; i++))
do 
  index=0
  ssh student@iGpu10 "ls -1v /data/$1/cam$i/*.bag" | while read entry; do
      if [ "${entry: -4}" != ".bag" ]; then
          continue
      fi
      echo "${entry}"
      python3 ./vision_toolkit/extract/read_bag_.py --bag "student@iGpu10:${entry}" --cam "$i" --save $1 --num $index
      ((index++))
  done
done
