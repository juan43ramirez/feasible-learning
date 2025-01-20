MAX_STEPS=5000
for tolerance in 0.000001 0.00001 0.0001 0.001 0.01 0.1
do
    CUDA_VISIBLE_DEVICES=1 python dpof.py --model_name_or_path=gpt2 --per_device_train_batch_size 4 --max_steps 5000 --learning_rate 1e-3 --gradient_accumulation_steps 1 --logging_steps 100 --eval_steps 500 --output_dir=dpo_anthropic_hh --optim rmsprop --warmup_steps 150 --report_to wandb --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=16 --lora_alpha=1 --loss_tolerance $tolerance --dual_lr 0.0
done