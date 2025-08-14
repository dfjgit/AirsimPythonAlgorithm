@echo off  
chcp 65001
set OUTPUT_DIR=.\packages  

if not exist %OUTPUT_DIR% (  
    mkdir %OUTPUT_DIR%  
)  

REM 先下载 setuptools 和 wheel  
pip download setuptools wheel --dest %OUTPUT_DIR%  
 
REM 导出项目的依赖包  
pip freeze > requirements.txt  

REM 下载所有依赖包及其依赖项  
pip download -r requirements.txt --dest %OUTPUT_DIR%  

@REM if exist requirements.txt (  
@REM     del requirements.txt  
@REM )  

echo 所有包已成功下载到 %OUTPUT_DIR% 目录中。  
pause 