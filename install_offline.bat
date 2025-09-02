@echo off

REM 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到Python安装。请先安装Python。
    pause
    exit /b 1
)

REM 创建dependencies目录（如果不存在）
if not exist "dependencies" mkdir "dependencies"

REM 从requirements.txt安装依赖项
cd /d "%~dp0"
echo 正在安装依赖项...
pip install --no-index --find-links=dependencies -r requirements.txt

if %errorlevel% neq 0 (
    echo 错误：依赖项安装失败。请确保dependencies目录包含所有必要的wheel文件。
    pause
    exit /b 1
)

REM 安装项目本身
echo 正在安装DroneServer项目...
python setup.py install

if %errorlevel% neq 0 (
    echo 错误：项目安装失败。
    pause
    exit /b 1
)

echo 安装成功！
pause