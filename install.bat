echo off
chcp 65001
echo python 3.6.0
echo python 环境安装中... 
pip install --no-index --find-links=.\packages -r requirements.txt
pause