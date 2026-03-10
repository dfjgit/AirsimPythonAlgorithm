@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
python train_with_airsim_improved.py --config configs/unified_train_config.json --total-timesteps 21600
pause

