export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/runwayml/11_multicontrol_train_None_all_noNorm_rerun1"

accelerate launch train_multicontrolnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./fs/nexus-projects/DroneHuman/jxchen/data/04_ev/runwayml/multicontrol_train_test/imgs/000120_t0.png" "/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/runwayml/multicontrol_train_test/imgs/000120_event.png" \
 --validation_prompt "" \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --num_train_epochs=2000 \
 --validation_steps=200 \
#  --push_to_hub
# --learning_rate=1e-5 \