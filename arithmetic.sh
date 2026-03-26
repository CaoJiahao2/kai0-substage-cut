python model_arithmetic/arithmetic.py \
    --config pi05_flatten_fold_normal_follow_paper_split2  \
    --data-path kai0_taskA_val.pkl \
    --checkpoints /magicflow/chenxuhua/Models/magiclab_works/kai0/pi05_flatten_fold_normal_follow_paper/flatten_fold_full_finetune_bs128_split1_0310/29999 /magicflow/chenxuhua/Models/magiclab_works/kai0/pi05_flatten_fold_normal_follow_paper_split2/flatten_fold_full_finetune_bs128_split2_0311/29999  /magicflow/chenxuhua/Models/magiclab_works/kai0/pi05_flatten_fold_normal_follow_paper_split3/flatten_fold_full_finetune_bs128_split3_0312/29999\
    --output /magicflow/chenxuhua/Models/magiclab_works/kai0/pi05_flatten_normal_Task_A_mixed \
    --optimize_method inverse_loss 