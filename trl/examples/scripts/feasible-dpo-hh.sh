for resilient_alpha in 1.0 100.0 0.01
do
    for dual_lr in 1.0 10.0 0.1
    do
        for tolerance in 0.8 0.5 0.0
        do
            python dpof.py --algorithm feasible --optim paged_adamw_32bit --dataset hh --train_epochs 10 --model_name_or_path=gpt2 --per_device_train_batch_size 6  --learning_rate 1e-3 --gradient_accumulation_steps 8 --logging_steps 100 --eval_steps 500 --output_dir=dpo_anthropic_hh --warmup_steps 150 --report_to wandb --bf16 --logging_first_step --no_remove_unused_columns --use_peft --lora_r=16 --lora_alpha=1 --loss_tolerance $tolerance --dual_lr $dual_lr --resilient_alpha $resilient_alpha
        done
    done
done