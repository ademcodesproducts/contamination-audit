@echo off
REM Run the full pilot pipeline end-to-end on Windows.
REM Usage: run_all.bat [max_pairs]
REM   run_all.bat          -- full run
REM   run_all.bat 200      -- quick test (judge only 200 pairs)
REM
REM Prerequisites:
REM   pip install -r requirements.txt

setlocal

set MAX_PAIRS=%1

echo === Step 1: Download data ===
python 01_download_data.py
if errorlevel 1 goto :error

echo.
echo === Step 2: 8-gram baseline (expect ~0 hits) ===
python 02_ngram_baseline.py
if errorlevel 1 goto :error

echo.
echo === Step 3A: Embedding retrieval ===
python 03_embedding_retrieval.py
if errorlevel 1 goto :error

echo.
echo === Step 3B: LLM judge ===
if "%MAX_PAIRS%"=="" (
    python 04_llm_judge.py --resume
) else (
    python 04_llm_judge.py --max-pairs %MAX_PAIRS% --resume
)
if errorlevel 1 goto :error

echo.
echo === Step 4: Analyze results ===
python 05_analyze_results.py
if errorlevel 1 goto :error

echo.
echo === Done! ===
echo Key outputs:
echo   results\pilot_report.md         -- summary with the 4 key numbers
echo   results\flagged_pairs.tsv       -- all CONTAMINATED/RELATED pairs
echo   results\precision_worksheet.tsv -- fill this in for manual validation
goto :eof

:error
echo.
echo ERROR: Step failed. Check output above.
exit /b 1
