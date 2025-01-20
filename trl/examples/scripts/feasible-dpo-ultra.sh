for resilient_alpha in 1.0
do
    for dual_lr in 1.0
    do
        for tolerance in 0.8 0.5 0.1
        do
            python dpof.py --gradient_checkpointing True --dataset ultra --train_epochs 10 --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora --algorithm feasible --per_device_train_batch_size 4 --learning_rate 1e-3 --gradient_accumulation_steps 1 --logging_steps 100 --eval_steps 500 --output_dir=dpo_anthropic_hh --optim paged_adamw_32bit --warmup_steps 150 --report_to wandb --bf16 --logging_first_step --no_remove_unused_columns --use_peft --load_in_4bit --lora_r 16 --loss_tolerance $tolerance --dual_lr $dual_lr --resilient_alpha $resilient_alpha
        done
    done
done