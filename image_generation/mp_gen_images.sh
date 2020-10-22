#! /bin/bash

NUM_PROCS=$1
START_IDX=$2
NUM_IMAGES=$3
OUTPUT=$4
NUM_IMAGES_PER_PROC=$((NUM_IMAGES / NUM_PROCS))
# START_SEED=5

# for SEED in {$START_SEED..$((START_SEED+NUM_PROCS))}
# for ((SEED=$START_SEED; SEED<$START_SEED + $NUM_PROCS; SEED++))
for ((IDX=$START_IDX; IDX<$((START_IDX + NUM_IMAGES)); IDX=$((IDX + NUM_IMAGES_PER_PROC))))
do
    $BLENDER_PATH --background --python render_images.py -- --num_images $NUM_IMAGES_PER_PROC \
                                                            --width 480 \
                                                            --height 320 \
                                                            --start_idx $IDX \
                                                            --output_image_dir="$OUTPUT/images" \
                                                            --output_depth_dir="$OUTPUT/depth"\
                                                            --output_scene_dir="$OUTPUT/scenes"\
                                                            --output_blend_dir="$OUTPUT/blendfiles"\
                                                            --output_cam_dir="$OUTPUT/cameras" \
                                                            --output_scene_file="$OUTPUT/CLEVR_scenes.json" \
                                                            --properties_json="data/properties_novel_only.json" \
                                                            &
                                                            # --contrastive-info-file="$OUTPUT/CLEVR_CE.json"\
                                                            # --contrastive \
                                                            # --all_views \
                                                            # --floating \
                                                            # --prototype="garlic" \
                                                            # --save_blendfiles=1 \
                                                            # --min_objects 1 \
                                                            # --max_objects 1 \
                                                            # --base_scene_blendfile="data/base_scene_full2.blend" \
    sleep 2
done

# SEED=10 ./gen_images.sh &
# sleep 1
# SEED=11 ./gen_images.sh &
# sleep 1
# SEED=12 ./gen_images.sh &
# sleep 1
# SEED=13 ./gen_images.sh &
# sleep 1
# SEED=14 ./gen_images.sh &
# sleep 1
# SEED=15 ./gen_images.sh &
