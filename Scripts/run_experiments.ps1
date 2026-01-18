# Script to run Baseline and PyraPatch experiments on the ETTh1 dataset

Write-Host "Starting Baseline Run on ETTh1 Dataset..."
python PatchTST_supervised/run_longExp.py --is_training 1 --root_path "C:\Users\NICK\PycharmProjects\PatchTST\PatchTST_supervised\dataset\" --data_path ETTh1.csv --model_id Baseline_ETTh1 --model PatchTST --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --e_layers 3 --d_model 128 --patch_len 16 --stride 8 --des 'Exp' --itr 1 --num_stages 1 --patience 5

Write-Host "Baseline Run Completed. Starting PyraPatch Run..."
python PatchTST_supervised/run_longExp.py --is_training 1 --root_path "C:\Users\NICK\PycharmProjects\PatchTST\PatchTST_supervised\dataset\" --data_path ETTh1.csv --model_id PyraPatch_ETTh1 --model PatchTST --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --e_layers 3 --d_model 128 --patch_len 8 --stride 4 --des 'Exp' --itr 1 --num_stages 3 --patience 10

Write-Host "All experiments completed."
