@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo           文件计数工具
echo ========================================
echo.

REM 设置默认目录为当前目录
set "target_dir=%~1"
if "%target_dir%"=="" set "target_dir=."

REM 检查目录是否存在
if not exist "%target_dir%" (
    echo 错误：目录 "%target_dir%" 不存在！
    pause
    exit /b 1
)

echo 正在扫描目录: %target_dir%
echo.
echo 开始递归扫描所有子文件夹...
echo.

REM 初始化计数器
set /a file_count=0
set /a dir_count=0
set /a top_level_files=0

REM 递归遍历目录中的所有文件和文件夹
for /r "%target_dir%" %%i in (*) do (
    set /a file_count+=1
    echo [文件] %%~nxi - %%~fi
)

REM 递归遍历目录中的所有子目录
for /r "%target_dir%" %%i in (.) do (
    if not "%%~fi"=="%target_dir%" (
        set /a dir_count+=1
        echo [目录] %%~nxi - %%~fi
    )
)

REM 统计最顶层文件夹内的文件数量
for %%i in ("%target_dir%\*") do (
    if not exist "%%i\" (
        set /a top_level_files+=1
    )
)

echo.
echo ========================================
echo 统计结果：
echo ========================================
echo 文件数量: %file_count%
echo 目录数量: %dir_count%
set /a total=%file_count%+%dir_count%
echo 总项目数: %total%
echo.
echo ========================================
echo 最顶层文件夹统计：
echo ========================================
echo 最顶层文件数量: %top_level_files%
echo.
echo 目录路径: %target_dir%
echo.
pause