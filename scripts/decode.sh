CUDA_VISIBLE_DEVICES=0 python decode_pipe.py \
    --load_state_dict ecspell_law_mlm+re/step-4000_f1-74.23.bin \
    --eval_on ecspell/test_law.txt \
    --output_dir decode_ecspell_lawmlm_d2c \
    --vocab wordlist/公文写作.txt \
    --max_seq_length 128 \
    --change_logits 0.85 \
    --max_range 15 \
    --calc_prd \