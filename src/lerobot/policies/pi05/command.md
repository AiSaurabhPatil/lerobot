#### Training LORA (first need to compute quantile statistics of the dataset )
```sh
uv run accelerate launch \
  --num_processes=2 \
  --multi_gpu \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --policy.path=.hf-cache/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.push_to_hub=false \
  --dataset.repo_id=ffw_bg2_v3 \
  --dataset.root=datasets/ffw_bg2_v3 \
  --batch_size=6 \
  --steps=20000 \
  --num_workers=6 \
  --prefetch_factor=4 \
  --eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --log_freq=50 \
  --wandb.enable=true \
  --wandb.project=lerobot-pi05 \
  --wandb.mode=online \
  --output_dir=outputs/train/pi05_lora_ffw_bg \
  --peft.method_type=LORA \
  --peft.r=32
```

### Qantile stats compute command  (for pi05 policy)
```sh
uv run python -c "from pathlib import Path; from lerobot.datasets import LeRobotDataset, write_stats; from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset; root=Path('/home/saurabh/Development/lerobot/datasets/ffw_bg2_v3'); ds=LeRobotDataset('local/ffw_bg2_v3', root=root); write_stats(compute_quantile_stats_for_dataset(ds), root)"
```



#### Inference (To open the enviroment in isaac sim)
```sh
/home/saurabh/isaac_sim/python.sh scripts/isaac_ros_usd_bridge.py \
  --fps 30 --host 127.0.0.1 --port 8765 \
  --action-stats-json datasets/quest3-acone/meta/stats.json \
  --max-joint-target-step 0.02
```

#### Inference (to run the inference engine)
```sh 
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=/mnt/storage/saurabh_data/pi05_checkpoint/012000/pretrained_model \
  --policy.num_inference_steps=5 \
  --robot.type=ros_isaac \
  --robot.bridge_host=127.0.0.1 \
  --robot.bridge_port=8765 \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --fps=30 \
  --duration=60 \
  --device=cuda \
  --task="sort nuts and bolts in different bins " \
  --return_to_initial_position=false
```





#### SmolVLA Training Command 
```sh
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=ffw_bg2_v3 \
  --dataset.root=datasets/ffw_bg2_v3 \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ffw_bg2_smolvla \
  --job_name=ffw_bg2_smolvla \
  --batch_size=8 \
  --steps=20000 \
  --eval_freq=-1 \
  --save_freq=5000 \
  --num_workers=8 \
  --rename_map='{"observation.images.head_camera":"observation.images.camera1","observation.images.left_wrist_camera":"observation.images.camera2","observation.images.right_wrist_camera":"observation.images.camera3"}'
```

#### SmolVLA Inference Command 
```sh 
## Terminal 1 
/home/saurabh/isaac_sim/python.sh scripts/isaac_ros_usd_bridge.py   --robot-config /home/saurabh/Development/quest3_streamer/config/robots/ffw_bg2.yaml   --dataset-root /home/saurabh/Development/lerobot/datasets/ffw_bg2_v2



## Terminal 2 
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=/home/saurabh/Development/lerobot/outputs/train/ffw_bg2_smolvla/checkpoints/020000/pretrained_model \
  --robot.type=ros_isaac \
  --robot.robot_config=/home/saurabh/Development/quest3_streamer/config/robots/ffw_bg2.yaml \
  --robot.bridge_host=127.0.0.1 \
  --robot.bridge_port=8765 \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --fps=30 \
  --duration=60 \
  --device=cuda \
  --task="put cube in tray" \
  --return_to_initial_position=false \
  --rename_map='{"observation.images.head_camera":"observation.images.camera1","observation.images.left_wrist_camera":"observation.images.camera2","observation.images.right_wrist_camera":"observation.images.camera3"}'
```