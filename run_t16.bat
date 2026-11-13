@echo off
REM ============================================================
REM  T-16 budget-constrained weight search — NORMAL RUN (launch only, no auto-run)
REM  Launches the production proxy->confirm search in a minimized
REM  background window (~1.5-2h at 12 workers) and streams to _t16_run.log.
REM  Safe to run any time; results are deterministic per --seed-base.
REM ============================================================
setlocal
cd /d "%~dp0"

REM ---- sanity checks (fail fast before spending ~2h) ----
if not exist "ai\target\release\selfplay.exe" (
    echo [ERROR] selfplay.exe not found at ai\target\release\selfplay.exe
    echo         Build it first:  cd /d "%~dp0ai" ^&^& cargo build --release
    exit /b 1
)
if not exist "weights_v0.json" (
    echo [ERROR] anchor weights_v0.json not found next to this bat.
    exit /b 1
)

set CMD=py -u tools_budget_search.py --candidates 40 --seed-base 0 --proxy-games 15 --proxy-depth 4 --confirm-games 50 --confirm-depth 6 --maxmove 480 --workers 12 --max-confirm 3 --out-dir .

start "T16 budget search" /min cmd /c "%CMD% > _t16_run.log 2>&1"

echo Launched in background (minimized window: "T16 budget search").
echo   Running:
echo     %CMD%
echo   Log:      _t16_run.log       (watch this for live progress)
echo   Results:  budget_search_log.csv
echo             budget_search_verdict.txt
echo             best_weights.json   (only if a candidate is ACCEPTED)
echo   Check the verdict when done:  type budget_search_verdict.txt
endlocal
exit /b 0
