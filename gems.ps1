<#
.SYNOPSIS
    EMShrimulator Studio launcher: setup, run, build, diagnose.

.DESCRIPTION
    One command that gets you from a fresh checkout to a working app.

      .\gems.ps1              start the desktop application
      .\gems.ps1 setup        create the venv and install everything
      .\gems.ps1 build        build the standalone folder and zip (optional)
      .\gems.ps1 doctor       check the install and print what is missing
      .\gems.ps1 test         run the test suite
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('run', 'setup', 'build', 'doctor', 'test')]
    [string]$Command = 'run',

    [string]$Open,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'
$VenvPyw = Join-Path $Root '.venv\Scripts\pythonw.exe'

$ProgressPreference = 'SilentlyContinue'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "  !!  $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "  XX  $msg" -ForegroundColor Red }

function Invoke-Native {
    # pip logs to stderr, which 'Stop' would treat as failure; trust the exit code instead.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$FailureMessage = 'command failed'
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $FilePath @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "$FailureMessage (exit code $code)" }
}

function Find-Python313 {
    # The CSXCAD/openEMS wheels are built for CPython 3.13 only.
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:ProgramFiles\Python313\python.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        $found = & $py.Source -3.13 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { return $found.Trim() }
    }
    return $null
}

function Get-OpenEMSDir {
    if ($env:OPENEMS_DIR -and (Test-Path $env:OPENEMS_DIR)) { return $env:OPENEMS_DIR }
    foreach ($c in @('C:\OpenEMS\openEMS', 'D:\OpenEMS\openEMS', "$env:ProgramFiles\openEMS")) {
        if (Test-Path (Join-Path $c 'openEMS.exe')) { return $c }
    }
    return $null
}

function Use-Environment {
    $openems = Get-OpenEMSDir
    if ($openems) { $env:OPENEMS_DIR = $openems }
    $env:PYTHONPATH = $Root
    $env:PYTHONIOENCODING = 'utf-8'
}

function Assert-Venv {
    if (-not (Test-Path $VenvPy)) {
        Write-Bad 'The virtual environment is missing.'
        Write-Host '     Run:  .\gems.ps1 setup'
        throw 'Not set up yet.'
    }
}

# ------------------------------------------------------------------- setup

function Invoke-Setup {
    Write-Step 'Setting up EMShrimulator Studio'

    $openems = Get-OpenEMSDir
    if (-not $openems) {
        Write-Bad 'openEMS was not found.'
        Write-Host '     Download openEMS for Windows: https://github.com/thliebig/openEMS-Project/releases'
        Write-Host '     Unzip it to C:\OpenEMS\openEMS, or set OPENEMS_DIR to the folder with openEMS.exe.'
        throw 'openEMS is required.'
    }
    Write-Ok "openEMS at $openems"
    $env:OPENEMS_DIR = $openems

    if ($Force -and (Test-Path (Join-Path $Root '.venv'))) {
        Write-Step 'Removing the existing venv (-Force)'
        Remove-Item -Recurse -Force (Join-Path $Root '.venv')
    }

    if (-not (Test-Path $VenvPy)) {
        $py = Find-Python313
        if (-not $py) {
            Write-Bad 'Python 3.13 was not found.'
            Write-Host '     The openEMS/CSXCAD wheels are built for CPython 3.13 specifically.'
            Write-Host '     Install it with:  winget install Python.Python.3.13'
            throw 'Python 3.13 is required.'
        }
        Write-Step "Creating the virtual environment with $py"
        Invoke-Native $py @('-m', 'venv', (Join-Path $Root '.venv')) 'venv creation failed'
    }
    Write-Ok "venv at $VenvPy"

    Write-Step 'Installing Python dependencies (this takes a few minutes the first time)'
    Invoke-Native $VenvPy @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet') 'upgrading pip failed'

    $wheelDir = Join-Path $openems 'python'
    $wheels = @()
    if (Test-Path $wheelDir) {
        $wheels = @(Get-ChildItem $wheelDir -Filter '*cp313*.whl' -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match 'csxcad|openems' } |
                    ForEach-Object { $_.FullName })
    }
    if ($wheels.Count -gt 0) {
        Write-Step "Installing the CSXCAD/openEMS wheels from $wheelDir"
        Invoke-Native $VenvPy (@('-m', 'pip', 'install', '--quiet') + $wheels) 'installing the CSXCAD/openEMS wheels failed'
    } else {
        Write-Warn2 "No cp313 wheels under $wheelDir - assuming CSXCAD/openEMS are already installed."
    }

    Invoke-Native $VenvPy @('-m', 'pip', 'install', '--quiet', '-r', (Join-Path $Root 'requirements.txt')) 'installing the Python dependencies failed'
    Write-Ok 'Python dependencies installed'

    Write-Step 'Installing the vendored gerber2ems'
    $env:SETUPTOOLS_SCM_PRETEND_VERSION = '0.1.0'
    Invoke-Native $VenvPy @('-m', 'pip', 'install', '--quiet', '-e', (Join-Path $Root 'vendor\gerber2ems-main'), '--no-deps') 'installing gerber2ems failed'
    Write-Ok 'gerber2ems installed (patched: gerbv replaced by gerbonara + resvg)'

    Write-Host ''
    Write-Ok 'Setup complete. Start the app with:  .\gems.ps1'
}

# ------------------------------------------------------------------ doctor

function Invoke-Doctor {
    Assert-Venv
    Use-Environment
    Write-Step 'Checking the install'
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    # Out-Host: printed, not returned - the function's only output must be the exit code.
    & $VenvPy -m EMShrimulator.core.diagnostics | Out-Host
    $code = $LASTEXITCODE
    & $VenvPy -c "import PySide6, vtkmodules, vtkmodules.qt.QVTKRenderWindowInteractor; print('  OK  Qt', PySide6.__version__, '| VTK', vtkmodules.__version__)" | Out-Host
    if ($LASTEXITCODE -ne 0) { $code = 1; Write-Bad 'The desktop GUI libraries are missing - run: .\gems.ps1 setup' }
    $ErrorActionPreference = $previous
    Write-Host ''
    if ($code -eq 0) { Write-Ok 'Everything checks out.' } else { Write-Bad 'Problems found - see above.' }
    return [int]$code
}

# --------------------------------------------------------------------- run

function Invoke-Run {
    Assert-Venv
    Use-Environment
    $argsList = @('-m', 'EMShrimulator')
    if ($Open) { $argsList += @('--open', $Open) }
    $exe = if (Test-Path $VenvPyw) { $VenvPyw } else { $VenvPy }
    Write-Step 'Starting EMShrimulator Studio (the first start loads the solver and 3D libraries, which can take a while)'
    Start-Process -FilePath $exe -ArgumentList $argsList -WorkingDirectory $Root
}

# ------------------------------------------------------------------- build

function Invoke-Build {
    Assert-Venv
    Use-Environment
    $argsList = @((Join-Path $Root 'packaging\build.py'))
    Invoke-Native $VenvPy $argsList 'the build failed'
}

function Invoke-Test {
    Assert-Venv
    Use-Environment
    $tests = @(Get-ChildItem (Join-Path $Root 'tests') -Filter 'test_*.py')
    # Files made of test_ functions need pytest; run as scripts they would only define them.
    $pytestFiles = @($tests | Where-Object { Select-String -Path $_.FullName -Pattern '^def test_' -Quiet })
    if ($pytestFiles.Count) {
        Invoke-Native $VenvPy (@('-m', 'pytest', '-q') + $pytestFiles.FullName) 'pytest tests failed'
    }
    $tests | Where-Object { $pytestFiles.FullName -notcontains $_.FullName } | ForEach-Object {
        Invoke-Native $VenvPy @($_.FullName) "$($_.Name) failed"
    }
}

# ------------------------------------------------------------------ dispatch

switch ($Command) {
    'setup'  { Invoke-Setup }
    'build'  { Invoke-Build }
    'doctor' { $n = Invoke-Doctor; exit ([int]($n -gt 0)) }
    'test'   { Invoke-Test }
    default  { Invoke-Run }
}
