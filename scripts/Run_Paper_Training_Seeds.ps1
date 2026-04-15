param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ddpg_apf", "pure_dqn")]
    [string]$Algorithm,

    [string[]]$Seeds = @("20260413", "20260414", "20260415"),

    [string]$StageName = "stage01",

    [int]$StageIndex = 1,

    [string]$ContinueModel = "",

    [switch]$UsePretrainedDqn
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$uiLang = [System.Environment]::GetEnvironmentVariable("AIRSIM_UI_LANG", "Process")
$quickSeeds = [System.Environment]::GetEnvironmentVariable("AIRSIM_QUICK_SEEDS", "Process")
$pythonExe = "python"
if (Test-Path (Join-Path $root "myvenv\Scripts\python.exe")) {
    $pythonExe = (Join-Path $root "myvenv\Scripts\python.exe")
}
elseif (Test-Path (Join-Path $root "..\..\myvenv\Scripts\python.exe")) {
    $pythonExe = (Resolve-Path (Join-Path $root "..\..\myvenv\Scripts\python.exe")).Path
}

function Invoke-SeededPython {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments,
        [hashtable]$Environment
    )

    $previous = @{}
    foreach ($key in $Environment.Keys) {
        $previous[$key] = [System.Environment]::GetEnvironmentVariable($key, "Process")
        [System.Environment]::SetEnvironmentVariable($key, $Environment[$key], "Process")
    }

    try {
        [System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")
        [System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
        & $pythonExe $ScriptPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            if ($uiLang -eq "zh") {
                throw "命令执行失败，退出码: $LASTEXITCODE"
            }
            throw "Command failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        foreach ($key in $Environment.Keys) {
            [System.Environment]::SetEnvironmentVariable($key, $previous[$key], "Process")
        }
    }
}

foreach ($seed in $Seeds) {
    Write-Host "============================================================"
    if ($uiLang -eq "zh") {
        Write-Host "算法: $Algorithm | Seed: $seed | 阶段: $StageName"
    }
    else {
        Write-Host "Algorithm: $Algorithm | Seed: $seed | Stage: $StageName"
    }
    Write-Host "============================================================"

    if ($Algorithm -eq "ddpg_apf") {
        $scriptPath = Join-Path $root "multirotor\DDPG_Weight\train_with_airsim_improved.py"
        $args = @("--seed", $seed)
        if ($ContinueModel) {
            $args += @("--continue-model", $ContinueModel)
        }
        Invoke-SeededPython `
            -ScriptPath $scriptPath `
            -Arguments $args `
            -Environment @{
                "TRAIN_SEED" = $seed
                "EXPERIMENT_ID" = "ddpg_apf_seed_$seed"
                "TRAIN_STAGE_NAME" = $StageName
                "TRAIN_STAGE_INDEX" = "$StageIndex"
            }
    }
    elseif ($Algorithm -eq "pure_dqn") {
        $scriptPath = Join-Path $root "multirotor\DQN_Movement\scripts\train_movement_with_airsim.py"
        $args = @("--seed", $seed)
        $usePretrained = if ($UsePretrainedDqn.IsPresent) { "1" } else { "0" }
        Invoke-SeededPython `
            -ScriptPath $scriptPath `
            -Arguments $args `
            -Environment @{
                "TRAIN_SEED" = $seed
                "EXPERIMENT_ID" = "pure_dqn_seed_$seed"
                "TRAIN_STAGE_NAME" = $StageName
                "TRAIN_STAGE_INDEX" = "$StageIndex"
                "USE_PRETRAINED" = $usePretrained
            }
    }
}
if (-not [string]::IsNullOrWhiteSpace($quickSeeds)) {
    $Seeds = $quickSeeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
