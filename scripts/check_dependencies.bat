@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cls

echo ============================================================
echo 检查系统依赖
echo ============================================================
echo.

REM 激活虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [1/2] 激活虚拟环境...
    call .venv\Scripts\activate.bat >nul 2>&1
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，使用系统Python
)
echo.

REM 检查并安装 python-socketio
echo [2/2] 检查 python-socketio...
REM 使用更可靠的方法：先检查 Python 是否可用
where python >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [错误] 未找到 Python，请先安装 Python
    echo.
    pause
    exit /b 1
)

REM 检查 socketio 是否已安装
python -c "import socketio" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo [提示] python-socketio 未安装，正在安装...
    python -m pip install "python-socketio[client]"
    if !ERRORLEVEL! NEQ 0 (
        echo [警告] pip 安装返回错误，但继续检查...
    )
    REM 等待安装完成
    timeout /t 2 /nobreak >nul
    REM 再次检查导入
    python -c "import socketio" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo [错误] python-socketio 安装失败或无法导入！
        echo [提示] 请手动运行: pip install python-socketio[client]
        echo [调试] 尝试显示 Python 路径...
        python -c "import sys; print(sys.executable)" 2>nul
        echo.
        pause
        exit /b 1
    ) else (
        echo [OK] python-socketio 安装成功
    )
) else (
    echo [OK] python-socketio 已安装
)
echo.

echo ============================================================
echo [成功] 所有依赖检查通过！
echo ============================================================
echo.
pause

