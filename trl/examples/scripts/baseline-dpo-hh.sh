for algorithm in "erm" "clamped"
do
    for tolerance in 0.1
    do
        CUDA_VISIBLE_DEVICES=1 python dpof.py --algorithm $algorithm --optim paged_adamw_32bit --dataset hh --train_epochs 3 --model_name_or_path=gpt2 --per_device_train_batch_size 4 --learning_rate 1e-3 --gradient_accumulation_steps 8 --logging_steps 100 --eval_steps 500 --output_dir=dpo_anthropic_hh --warmup_steps 150 --report_to wandb --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=16 --lora_alpha=1 --loss_tolerance $tolerance
    done
done