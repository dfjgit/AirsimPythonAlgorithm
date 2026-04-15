@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo 系统运行（固定权重）
echo ============================================================
echo.

set "AIRSIM_DRONE_COUNT=3"
if not "%AIRSIM_QUICK_DRONES%"=="" set "AIRSIM_DRONE_COUNT=%AIRSIM_QUICK_DRONES%"

REM 激活虚拟环境
echo [1/2] 激活 Python 虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，将使用系统 Python
)
echo.

REM 显示配置信息
echo ============================================================
echo 运行配置
echo ============================================================
echo 模式: 固定权重
echo 无人机数量: %AIRSIM_DRONE_COUNT%
echo ============================================================
echo.

REM 关键修改：先切换到AlgorithmServer.py所在的目录
echo [2/2] 切换到运行目录并启动算法服务器...
cd /d "%~dp0..\multirotor"  :: 切换到multirotor目录(AlgorithmServer.py在这里)

REM 直接用脚本名运行(此时是当前目录的短路径，无空格问题)
python AlgorithmServer.py --drones %AIRSIM_DRONE_COUNT%

echo.
echo ============================================================
echo 系统已退出
echo ============================================================
echo.
pause
