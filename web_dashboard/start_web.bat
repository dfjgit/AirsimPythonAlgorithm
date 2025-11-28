@echo off
chcp 65001 >nul
echo ========================================
echo 启动 Web 控制台服务器
echo ========================================
echo.

cd /d %~dp0

REM 检查 Node.js 是否安装
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Node.js！
    echo 请先安装 Node.js: https://nodejs.org/
    pause
    exit /b 1
)

REM 检查依赖是否已安装
if not exist "node_modules" (
    echo [提示] 首次运行，正在安装依赖...
    call npm install
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败！
        pause
        exit /b 1
    )
    echo.
)

echo [信息] 启动 Web 服务器...
echo [信息] 访问地址: http://localhost:3000
echo [信息] 按 Ctrl+C 停止服务器
echo.
echo ========================================
echo.

node server.js

pause

