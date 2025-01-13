@echo off

REM Set the build directory
set BUILD_DIR=build

REM Set the configuration (Release or Debug)
set CONFIG=Release

REM Step 1: Create the build directory if it doesn't exist
if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

REM Step 2: Navigate to the build directory
cd %BUILD_DIR%

REM Step 3: Run CMake to configure the project
echo Running CMake configuration...

:: 
:: IMPORTANT: Pass the vcpkg toolchain file AND specify x64 (if using x64-windows).
::
cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE="C:/Users/jonat/Documents/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
    -DCMAKE_BUILD_TYPE=%CONFIG%

if %errorlevel% neq 0 (
    echo CMake configuration failed!
    exit /b %errorlevel%
)

REM Step 4: Build the project
echo Building the project...
cmake --build . --config %CONFIG%

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

REM Step 5: Run the executable
echo Running the program...
if exist .\%CONFIG%\MyCudaProgram.exe (
    .\%CONFIG%\MyCudaProgram.exe
) else (
    echo Executable not found!
    exit /b 1
)

REM Step 6: Exit the script
cd ..
echo Build and run completed successfully!
exit /b 0

