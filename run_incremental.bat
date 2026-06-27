@echo off
REM ============================================================
REM FinRAG incremental crawl + ingest (entry for schtasks)
REM   download_data.py --incremental : watermark-based (crawl_state.json),
REM   each stock from last_run-7d to today, info_code dedup + quality gate,
REM   then --ingest runs ingest.py (fine-tuned bge); new reports auto-listed.
REM NOTE: keep this file PURE ASCII -- cmd.exe parses .bat in the OEM codepage
REM       (GBK on zh-CN Windows); any non-ASCII here breaks parsing.
REM Register (weekly Mon 03:00):
REM   schtasks /create /tn "FinRAG_Incremental" /tr "<full path to this bat>" /sc weekly /d MON /st 03:00 /f
REM Test now : run_incremental.bat
REM Query    : schtasks /query /tn "FinRAG_Incremental"
REM Delete   : schtasks /delete /tn "FinRAG_Incremental" /f
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "data\logs" mkdir "data\logs"
echo. >> "data\logs\crawl_incremental.log"
echo [%date% %time%] ====== incremental crawl+ingest START ====== >> "data\logs\crawl_incremental.log"
"C:\Users\23016\anaconda3\envs\py_312\python.exe" download_data.py --incremental --ingest >> "data\logs\crawl_incremental.log" 2>&1
echo [%date% %time%] ====== DONE (exit %errorlevel%) ====== >> "data\logs\crawl_incremental.log"
