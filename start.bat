@echo off
setlocal EnableDelayedExpansion
title Coder LLM Training Pipeline

echo.
echo ============================================================
echo   Coder LLM Training Pipeline
echo   Model: ~42M params  ^|  GPU: RTX 3050 6GB
echo ============================================================
echo.

:: Change to the project directory (same folder as this .bat file)
cd /d "%~dp0"

:: ============================================================
:: STEP 0 — Check Python is available
:: ============================================================
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Is your conda/venv environment active?
    echo         Run:  conda activate llm-train
    echo         or:   llm-train\Scripts\activate
    pause
    exit /b 1
)

:: Check PyTorch + CUDA
echo [CHECK] Verifying GPU and libraries...
python -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}  GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
if errorlevel 1 (
    echo [ERROR] PyTorch not installed or environment not active.
    echo         Run:  pip install torch --index-url https://download.pytorch.org/whl/cu124
    pause
    exit /b 1
)
echo.

:: ============================================================
:: STEP 1 — Download data  (skip if data\raw already has files)
:: ============================================================
echo [STEP 1/5] Checking training data...
set DATA_OK=0
for /f %%i in ('dir /s /b "data\raw\*.py" "data\raw\*.js" "data\raw\*.ts" 2^>nul ^| find /c /v ""') do set FILE_COUNT=%%i
if !FILE_COUNT! GTR 100 (
    echo          Found !FILE_COUNT! code files in data\raw\ -- skipping download.
    set DATA_OK=1
) else (
    echo          Not enough data found. Downloading codeparrot-small + CodeSearchNet...
    python scripts/download_data.py --dataset all-small --max_gb 2
    if errorlevel 1 (
        echo [ERROR] Data download failed.
        pause
        exit /b 1
    )
)
echo.

:: ============================================================
:: STEP 2 — Train tokenizer  (skip if models\tokenizer exists)
:: ============================================================
echo [STEP 2/5] Checking tokenizer...
if exist "models\tokenizer\tokenizer.json" (
    echo          Tokenizer already trained at models\tokenizer\ -- skipping.
) else (
    echo          Training BPE tokenizer on code files...
    python scripts/train_tokenizer.py
    if errorlevel 1 (
        echo [ERROR] Tokenizer training failed.
        pause
        exit /b 1
    )
)
echo.

:: ============================================================
:: STEP 3 — Preprocess data  (skip if data\processed\train.bin exists)
:: ============================================================
echo [STEP 3/5] Checking preprocessed data...
if exist "data\processed\train.bin" (
    echo          Preprocessed shards found at data\processed\ -- skipping.
) else (
    echo          Tokenising code files into binary shards...
    echo          (Using --num_workers 0 for Windows compatibility)
    python scripts/preprocess_data.py --num_workers 0
    if errorlevel 1 (
        echo [ERROR] Preprocessing failed.
        pause
        exit /b 1
    )
)
echo.

:: ============================================================
:: STEP 4 — Sanity check: print dataset stats
:: ============================================================
echo [STEP 4/5] Dataset info:
python -c "
import json, os
p = 'data/processed/meta.json'
if os.path.exists(p):
    m = json.load(open(p, encoding='utf-8'))
    print(f'  Vocab size   : {m[\"vocab_size\"]:,}')
    print(f'  Train tokens : {m[\"train_tokens\"]:,}')
    print(f'  Val tokens   : {m[\"val_tokens\"]:,}')
    print(f'  Dtype        : {m[\"dtype\"]}')
else:
    print('  meta.json not found')
"
echo.

:: ============================================================
:: STEP 5 — Train
:: ============================================================
echo [STEP 5/5] Starting training...
echo.
echo   Config : configs\train_config.yaml
echo   Model  : ~42M params  ^(hidden=512, layers=8, heads=8^)
echo   Seq len: 1024 tokens
echo   Batch  : 2 x 16 accum = 32 effective sequences per step
echo   Steps  : 50,000  ^(~24 hrs on RTX 3050^)
echo.
echo   Checkpoints saved every 2000 steps to models\checkpoints\
echo   Press Ctrl+C at any time to stop. Resume with:
echo     python scripts\train.py --resume models\checkpoints\step_NNNNNNN
echo.
echo ============================================================
echo   Training log  ^(starting in 5 seconds...^)
echo ============================================================
echo.

timeout /t 5 /nobreak >nul

python scripts/train.py
if errorlevel 1 (
    echo.
    echo [ERROR] Training exited with an error.
    echo         Check the output above for details.
    echo.
    echo   Common fixes:
    echo     OOM  ^(out of memory^)  -- edit configs\train_config.yaml:
    echo                                 batch_size: 1
    echo                                 gradient_accumulation_steps: 32
    echo     fp16 NaN loss        -- lower learning_rate to 1.0e-4
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Training complete!
echo ============================================================
echo.
echo   Generate code with:
echo     python scripts\generate.py --checkpoint models\checkpoints\step_XXXXXXX --interactive
echo.
pause
