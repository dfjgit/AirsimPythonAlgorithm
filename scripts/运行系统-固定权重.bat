@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo 启动系统 - 固定权重模式
echo ============================================================
echo.

REM 激活虚拟环境
echo [1/2] 激活Python虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，使用系统Python
)
echo.

REM 显示配置信息
echo ============================================================
echo 运行配置
echo ============================================================
echo 模式: 固定权重
echo 无人机数量: 3
echo ============================================================
echo.

REM 关键修改：先切换到AlgorithmServer.py所在的目录
echo [2/2] 切换到脚本目录并启动算法服务器....
cd /d "%~dp0..\multirotor"  :: 切换到multirotor目录（AlgorithmServer.py在这里）

REM 直接用脚本名运行（此时是当前目录的短路径，无空格问题）
python AlgorithmServer.py --drones 3

echo.
echo ============================================================
echo 程序已退出
echo ============================================================
echo.
pause