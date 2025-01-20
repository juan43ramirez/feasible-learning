epochs=20
tolerance=0.3
dual_lr=1.0
resilient_alpha=1.0
for algorithm in "erm" # "feasible" "clamped-erm"
do
    CUDA_VISIBLE_DEVICES=1 python dpof.py --evaluation_strategy epoch --sanity_check true --lr_scheduler_type "cosine" --beta 0.1 --max_prompt_length 1024 --max_length 1536  --learning_rate 5e-5 --algorithm ${algorithm} --optim paged_adamw_32bit --dataset orca --train_epochs $epochs --model_name_or_path=stabilityai/stablelm-zephyr-3b --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --logging_steps 100 --output_dir=dpo_intel --warmup_steps 200 --report_to wandb --bf16 --load_in_4bit --logging_first_step --no_remove_unused_columns --use_peft --lora_r=8 --lora_alpha=16 --loss_tolerance $tolerance --dual_lr $dual_lr --resilient_alpha $resilient_alpha
done