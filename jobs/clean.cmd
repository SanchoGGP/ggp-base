@echo off
call forfiles /P ..\logs /D -7 /C "cmd /c del @path"
call forfiles /P %USERPROFILE%\ggp-saved-matches /D -7 /C "cmd /c del @path"
for /d %%i in (..\data\games\1*) do rd /s /q %%i
