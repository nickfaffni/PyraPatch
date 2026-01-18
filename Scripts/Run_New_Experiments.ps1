# Comprehensive Experiment Runner for PyraPatch vs Baseline
# This script runs all feasible datasets while skipping heavy ones.
# Fixed: Long lines are now split to prevent copy-paste syntax errors.

$root_path = "C:\Users\NICK\PycharmProjects\PatchTST\PatchTST_supervised\dataset\"

# Define the datasets configurations
$experiments = @(
    @{name="ETTh1";    file="ETTh1.csv";         enc=7;  data="ETTh1"},
    @{name="ETTh2";    file="ETTh2.csv";         enc=7;  data="ETTh2"},
    @{name="ETTm1";    file="ETTm1.csv";         enc=7;  data="ETTm1"},
    @{name="ETTm2";    file="ETTm2.csv";         enc=7;  data="ETTm2"},
    @{name="Weather";  file="weather.csv";       enc=21; data="custom"},
    @{name="Exchange"; file="exchange_rate.csv"; enc=8;  data="custom"}
)

Write-Host "=== STARTING COMPREHENSIVE BENCHMARK SUITE ===" -ForegroundColor Cyan
Write-Host "Skipping Electricity/Traffic to ensure stability." -ForegroundColor Yellow
Write-Host "Added memory cleanup between runs." -ForegroundColor Yellow

foreach ($exp in $experiments) {
    $d_name = $exp.name
    $d_file = $exp.file
    $d_enc = $exp.enc
    $d_type = $exp.data

    Write-Host "`n========================================================"
    Write-Host "Processing Dataset: $d_name" -ForegroundColor Green
    Write-Host "========================================================"

    try {
        # 1. BASELINE RUN
        Write-Host "  > Running Baseline ($d_name)..."

        # Build command in parts to avoid line-wrap errors
        $cmdBase = "python PatchTST_supervised/run_longExp.py --is_training 1 "
        $cmdBase += "--root_path '$root_path' --data_path $d_file "
        $cmdBase += "--model_id Baseline_$d_name --model PatchTST --data $d_type --features M "
        $cmdBase += "--seq_len 336 --pred_len 96 --enc_in $d_enc --e_layers 3 --d_model 128 "
        $cmdBase += "--patch_len 16 --stride 8 --des 'Exp' --itr 1 --num_stages 1 --patience 3 "
        $cmdBase += "--batch_size 16 --num_workers 0"

        Invoke-Expression $cmdBase

        # Memory Cleanup
        [System.GC]::Collect()
        Start-Sleep -Seconds 2

        # 2. PYRAPATCH RUN (Ours)
        Write-Host "  > Running PyraPatch ($d_name)..."

        # Build command in parts
        $cmdPyra = "python PatchTST_supervised/run_longExp.py --is_training 1 "
        $cmdPyra += "--root_path '$root_path' --data_path $d_file "
        $cmdPyra += "--model_id PyraPatch_$d_name --model PatchTST --data $d_type --features M "
        $cmdPyra += "--seq_len 336 --pred_len 96 --enc_in $d_enc --e_layers 3 --d_model 128 "
        $cmdPyra += "--patch_len 8 --stride 4 --des 'Exp' --itr 1 --num_stages 3 --patience 3 "
        $cmdPyra += "--batch_size 16 --num_workers 0"

        Invoke-Expression $cmdPyra

        Write-Host "  completed $d_name" -ForegroundColor Green

        # Memory Cleanup and Cooldown
        [System.GC]::Collect()
        Start-Sleep -Seconds 3
    }
    catch {
        Write-Host "  Failed processing $d_name. Skipping to next..." -ForegroundColor Red
        Write-Host $_.Exception.Message
    }
}

Write-Host "`n=== All experiments finished! ===" -ForegroundColor Cyan