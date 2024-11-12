
for entry in "/lab/tmpig23b/navisim/data/bag_dump/$1/"*; do
    #python /lab/tmpig10c/kiran/nerf/GNerf/gaussian-splatting/render.py -m "$entry"
    echo "$entry"
    python ./extract_and_gpt4.py "$entry"
  done
