@echo off
chcp 65001 >nul
cls
echo ============================================================
echo    训练数据可视化分析工具
echo ============================================================
echo.
echo 功能说明：
echo   - 自动发现并分析所有训练数据
echo   - 支持 Crazyflie 实体无人机训练日志
echo   - 支持 DataCollector 扫描数据
echo   - 生成完整的分析报告和图表
echo.
echo ============================================================
echo.

:menu
echo.
echo 请选择分析模式：
echo.
echo   [1] 自动分析所有数据 (推荐)
echo   [2] 分析 Crazyflie 训练日志目录
echo   [3] 分析扫描数据目录
echo   [4] 分析指定文件
echo   [0] 返回主菜单
echo.
set /p choice=请输入选项 (0-4): 

if "%choice%"=="1" goto auto_analyze
if "%choice%"=="2" goto analyze_crazyflie
if "%choice%"=="3" goto analyze_scan
if "%choice%"=="4" goto analyze_file
if "%choice%"=="0" goto end

echo.
echo 无效的选项，请重新选择
timeout /t 2 >nul
goto menu

:auto_analyze
cls
echo ============================================================
echo 自动分析所有数据
echo ============================================================
echo.
echo 正在扫描数据目录...
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

python multirotor\Algorithm\visualize_training_data.py --auto --out analysis_results

echo.
echo ============================================================
echo 分析完成！
echo 结果保存在: %cd%\analysis_results
echo ============================================================
echo.

REM 询问是否打开结果目录
set /p open_dir=是否打开结果目录? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:analyze_crazyflie
cls
echo ============================================================
echo 分析 Crazyflie 训练日志
echo ============================================================
echo.
echo 正在分析 Crazyflie 训练数据...
echo 目录: multirotor\DDPG_Weight\crazyflie_logs
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

python multirotor\Algorithm\visualize_training_data.py --dir multirotor\DDPG_Weight\crazyflie_logs --out analysis_results

echo.
echo ============================================================
echo 分析完成！
echo 结果保存在: %cd%\analysis_results
echo ============================================================
echo.

REM 询问是否打开结果目录
set /p open_dir=是否打开结果目录? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:analyze_scan
cls
echo ============================================================
echo 分析扫描数据
echo ============================================================
echo.
echo 正在分析扫描数据...
echo 目录: multirotor\DDPG_Weight\airsim_training_logs
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

python multirotor\Algorithm\visualize_scan_csv.py --csv-dir multirotor\DDPG_Weight\airsim_training_logs --out analysis_results

echo.
echo ============================================================
echo 分析完成！
echo 结果保存在: %cd%\analysis_results
echo ============================================================
echo.

REM 询问是否打开结果目录
set /p open_dir=是否打开结果目录? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:analyze_file
cls
echo ============================================================
echo 分析指定文件
echo ============================================================
echo.
echo 请拖拽文件到此窗口，或输入文件路径：
echo 支持的格式: .json, .csv
echo.
set /p file_path=文件路径: 

if "%file_path%"=="" (
    echo.
    echo 未输入文件路径
    timeout /t 2 >nul
    goto menu
)

REM 移除引号
set file_path=%file_path:"=%

REM 检查文件是否存在
if not exist "%file_path%" (
    echo.
    echo 文件不存在: %file_path%
    pause
    goto menu
)

echo.
echo 正在分析文件...
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

REM 判断文件类型
echo %file_path% | findstr /i ".json" >nul
if %ERRORLEVEL% EQU 0 (
    python multirotor\Algorithm\visualize_training_data.py --json "%file_path%" --out analysis_results
) else (
    python multirotor\Algorithm\visualize_training_data.py --csv "%file_path%" --out analysis_results
)

echo.
echo ============================================================
echo 分析完成！
echo 结果保存在: %cd%\analysis_results
echo ============================================================
echo.

REM 询问是否打开结果目录
set /p open_dir=是否打开结果目录? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:end
echo.
echo 返回主菜单...
timeout /t 1 >nul
exit /b
