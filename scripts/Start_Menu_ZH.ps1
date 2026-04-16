[CmdletBinding()]
param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
Set-Location -LiteralPath $RepoRoot

if (-not $env:AIRSIM_RUNTIME_LOG_MODE) {
    $env:AIRSIM_RUNTIME_LOG_MODE = "user"
}

function Normalize-CrlfBytes {
    param([byte[]]$Bytes)

    $latin1 = [System.Text.Encoding]::GetEncoding(28591)
    $text = $latin1.GetString($Bytes)
    $normalized = $text.Replace("`r`n", "`n").Replace("`r", "`n").Replace("`n", "`r`n")
    return $latin1.GetBytes($normalized)
}

function Normalize-BatchFiles {
    $batchFiles = Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Extension -in @(".bat", ".cmd")
    }

    foreach ($batchFile in $batchFiles) {
        try {
            $original = [System.IO.File]::ReadAllBytes($batchFile.FullName)
            $normalized = Normalize-CrlfBytes -Bytes $original
        } catch {
            continue
        }

        $hasChanged = $original.Length -ne $normalized.Length
        if (-not $hasChanged) {
            $hasChanged = [System.BitConverter]::ToString($original) -ne [System.BitConverter]::ToString($normalized)
        }

        if ($hasChanged) {
            [System.IO.File]::WriteAllBytes($batchFile.FullName, $normalized)
        }
    }
}

function Get-PythonExe {
    $candidates = @(
        (Join-Path $RepoRoot "myvenv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return "python"
}

function Pause-IfNeeded {
    param(
        [string]$Prompt = "按回车继续..."
    )

    if ($env:AIRSIM_TEST_NO_PAUSE -eq "1") {
        return
    }

    Read-Host $Prompt | Out-Null
}

function Clear-QuickOverrides {
    foreach ($name in @(
        "AIRSIM_QUICK_DRONES",
        "AIRSIM_QUICK_DDPG_TIMESTEPS",
        "AIRSIM_QUICK_DQN_TIMESTEPS",
        "AIRSIM_QUICK_HRL_TIMESTEPS",
        "AIRSIM_QUICK_APF_BASELINE_EPISODES",
        "AIRSIM_QUICK_BENCHMARK_EPISODES",
        "AIRSIM_QUICK_VISUALIZATION",
        "AIRSIM_QUICK_SEEDS"
    )) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

function Collect-QuickConfig {
    param(
        [string]$Profile
    )

    if ([string]::IsNullOrWhiteSpace($Profile)) {
        return 0
    }

    if ($env:AIRSIM_TEST_SKIP_QUICK_CONFIG -eq "1") {
        return 0
    }

    Clear-QuickOverrides

    $outputFile = Join-Path $env:TEMP ("airsim_quick_config_{0}_{1}.env" -f (Get-Random), (Get-Random))
    if (Test-Path -LiteralPath $outputFile) {
        Remove-Item -LiteralPath $outputFile -Force -ErrorAction SilentlyContinue
    }

    $pythonExe = Get-PythonExe
    & $pythonExe `
        (Join-Path $RepoRoot "scripts\start_quick_config_helper.py") `
        --schema (Join-Path $RepoRoot "scripts\start_quick_config_schema.json") `
        --profile $Profile `
        --output $outputFile `
        --lang zh
    $exitCode = $LASTEXITCODE

    if (Test-Path -LiteralPath $outputFile) {
        foreach ($line in Get-Content -LiteralPath $outputFile -Encoding UTF8) {
            if ($line -match "^(?<name>[^=]+)=(?<value>.*)$") {
                Set-Item -Path ("Env:{0}" -f $Matches.name) -Value $Matches.value
            }
        }
        Remove-Item -LiteralPath $outputFile -Force -ErrorAction SilentlyContinue
    }

    return $exitCode
}

function Invoke-AsciiBatch {
    param(
        [string]$RelativePath,
        [string[]]$Arguments = @()
    )

    $scriptPath = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        Write-Host ""
        Write-Host "[错误] 未找到脚本: $RelativePath"
        Pause-IfNeeded
        return 1
    }

    $previousLang = $env:AIRSIM_UI_LANG
    $env:AIRSIM_UI_LANG = "en"
    try {
        & $scriptPath @Arguments
        return $LASTEXITCODE
    } finally {
        if ($null -eq $previousLang) {
            Remove-Item "Env:AIRSIM_UI_LANG" -ErrorAction SilentlyContinue
        } else {
            $env:AIRSIM_UI_LANG = $previousLang
        }
    }
}

function Query-LatestWorkflow {
    param(
        [string]$WorkflowType
    )

    if ([string]::IsNullOrWhiteSpace($WorkflowType)) {
        return $null
    }

    $pythonExe = Get-PythonExe
    $args = @(
        (Join-Path $RepoRoot "multirotor\Algorithm\paper_workflow_orchestrator.py"),
        "--workflow", $WorkflowType,
        "--query-latest-resumable"
    )
    if ($env:AIRSIM_WORKFLOW_WORKSPACE_ROOT) {
        $args += @("--workspace-root", $env:AIRSIM_WORKFLOW_WORKSPACE_ROOT)
    }

    $output = & $pythonExe @args
    if ($LASTEXITCODE -ne 0 -or -not $output) {
        return $null
    }

    $firstLine = @($output)[0]
    $parts = $firstLine -split "\|", 3
    if ($parts.Count -lt 1 -or [string]::IsNullOrWhiteSpace($parts[0])) {
        return $null
    }

    return [pscustomobject]@{
        Result = $parts[0]
        Status = if ($parts.Count -ge 2) { $parts[1] } else { "" }
        Phase  = if ($parts.Count -ge 3) { $parts[2] } else { "" }
    }
}

function Run-PaperWorkflow {
    param(
        [string]$WorkflowType,
        [string]$Mode
    )

    if ([string]::IsNullOrWhiteSpace($WorkflowType)) {
        return 1
    }

    if ($env:AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE) {
        Add-Content -LiteralPath $env:AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE -Value "$WorkflowType|$Mode"
        return 0
    }

    $args = @("--workflow", $WorkflowType)
    if ($Mode -eq "resume") {
        $args += "--resume-latest"
    }
    if ($env:AIRSIM_WORKFLOW_WORKSPACE_ROOT) {
        $args += @("--workspace-root", $env:AIRSIM_WORKFLOW_WORKSPACE_ROOT)
    }

    return (Invoke-AsciiBatch "scripts\Run_Paper_Workflow.bat" $args)
}

function Resolve-WorkflowMode {
    param(
        [string]$WorkflowType
    )

    $query = Query-LatestWorkflow $WorkflowType
    if ($null -ne $query) {
        Write-Host "检测到未完成的 workflow："
        Write-Host "  路径: $($query.Result)"
        if ($query.Status) {
            Write-Host "  状态: $($query.Status)"
        }
        if ($query.Phase) {
            Write-Host "  当前阶段: $($query.Phase)"
        }
        Write-Host ""
        Write-Host "可选操作："
        Write-Host "  [C] 继续当前实验"
        Write-Host "  [N] 新建实验并从头执行"
        Write-Host "  [Q] 返回主菜单"

        $workflowAction = if ($env:AIRSIM_TEST_WORKFLOW_ACTION) {
            $env:AIRSIM_TEST_WORKFLOW_ACTION
        } else {
            Read-Host "请选择操作 (C/N/Q)："
        }

        switch -Regex ($workflowAction) {
            '^(?i)C$' { return "resume" }
            '^(?i)N$' { return "new" }
            '^(?i)Q$' { return $null }
            default {
                Write-Host ""
                Write-Host "当前输入无效，请重新选择。"
                Start-Sleep -Seconds 2
                return "__invalid__"
            }
        }
    }

    $confirm = Read-Host "请输入 Y 继续执行，输入其它任意键返回主菜单："
    if ($confirm -ine "Y") {
        return $null
    }

    return "new"
}

function Show-ChineseMenu {
    Clear-Host
    Write-Host "============================================================"
    Write-Host "   AirSim 无人机仿真平台 - 控制台"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "=== 系统运行 ==="
    Write-Host "  [1] 启动系统（固定权重）"
    Write-Host "  [2] 启动系统（DDPG 权重预测）"
    Write-Host "  [3] 启动系统（DQN 控制，预留）"
    Write-Host ""
    Write-Host "=== DDPG+APF 训练 ==="
    Write-Host "  [4] 启动 DDPG+APF 训练（AirSim，新模型）"
    Write-Host "  [5] 继续 DDPG+APF 训练（AirSim）"
    Write-Host "  [6] 执行 DDPG+APF 训练（实体日志离线）"
    Write-Host "  [E] 执行 DDPG+APF 训练（实体无人机单轮在线）"
    Write-Host ""
    Write-Host "=== DQN 控制训练 ==="
    Write-Host "  [7] 启动 DQN 控制训练（AirSim，新模型）"
    Write-Host "  [8] 继续 DQN 控制训练（AirSim）"
    Write-Host "  [R] 重新执行当前 stage02 训练"
    Write-Host "  [H] 启动分层 DQN 训练（离线 / Mock）"
    Write-Host "  [F] 启动分层 DQN 训练（AirSim 融合）"
    Write-Host "  [D] 验证 DQN 控制模型"
    Write-Host ""
    Write-Host "=== 结果分析 ==="
    Write-Host "  [A] 生成可视化分析结果"
    Write-Host "  [B] 生成 DDPG 与 DQN 对比分析"
    Write-Host ""
    Write-Host "=== 实验工作流 ==="
    Write-Host "  [M] 四组统一仿真对比阶段"
    Write-Host "  [N] 虚实两阶段实验工作流 (Virtual-Real Two-Stage Workflow)"
    Write-Host ""
    Write-Host "=== 四组论文实验 ==="
    Write-Host "  [G] 执行四组仿真评测（冻结策略）/ Benchmark"
    Write-Host "  [I] 生成四组主结果分析"
    Write-Host "  [J] 生成 Family 维度对比分析"
    Write-Host "  [K] 执行论文多 Seed 训练（DDPG+APF）"
    Write-Host "  [L] 执行论文多 Seed 训练（Pure DQN）"
    Write-Host ""
    Write-Host "=== 系统维护 ==="
    Write-Host "  [C] 清理训练与分析产出"
    Write-Host ""
    Write-Host "=== 平台信息 ==="
    Write-Host "  [9] 查看平台信息"
    if ($env:AIRSIM_RUNTIME_LOG_MODE -ieq "detail") {
        Write-Host "  当前运行时日志模式: 详细模式"
    } else {
        Write-Host "  当前运行时日志模式: 用户模式"
    }
    Write-Host "  [T] 切换运行时日志模式（当前会话）"
    Write-Host "  [0] 退出系统"
    Write-Host ""
    Write-Host "============================================================"
    Write-Host ""
}

function Show-Info {
    Clear-Host
    Write-Host "============================================================"
    Write-Host "   平台信息"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "核心目录:"
    Write-Host "  - multirotor\AlgorithmServer.py"
    Write-Host "  - multirotor\Algorithm\"
    Write-Host "  - multirotor\DDPG_Weight\"
    Write-Host "  - multirotor\DQN_Movement\"
    Write-Host "  - multirotor\benchmark_registry.json"
    Write-Host "  - docs\FOUR_GROUP_BENCHMARK_WORKFLOW_ZH.md"
    Write-Host ""
    Write-Host "常用入口:"
    Write-Host "  - start.bat"
    Write-Host "  - scripts\Run_Four_Group_Benchmark.bat"
    Write-Host "  - scripts\Analyze_Four_Group_Benchmark.bat"
    Write-Host "  - scripts\Analyze_Family_Comparisons.bat"
    Write-Host "  - scripts\Run_Paper_Training_Seeds.ps1"
    Write-Host "  - docs\START_QUICK_CONFIG_ZH.md"
    Write-Host ""
    Write-Host "运行环境:"
    $pythonExe = Get-PythonExe
    try {
        & $pythonExe --version
        Write-Host "[OK] Python 环境可用"
    } catch {
        Write-Host "[提示] 未检测到可用的 Python 环境，请检查 myvenv 或 .venv"
    }
    Write-Host ""
    Pause-IfNeeded
}

function Confirm-Delete {
    param(
        [string]$TargetDir,
        [string]$TargetDesc
    )

    if ([string]::IsNullOrWhiteSpace($TargetDir)) {
        return
    }

    Clear-Host
    Write-Host "============================================================"
    Write-Host "删除确认: $TargetDesc"
    Write-Host "------------------------------------------------------------"
    Write-Host "路径: $TargetDir"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "[警告] 该操作不可撤销！"
    Write-Host ""
    $confirm = Read-Host "请输入 YES 确认删除，输入其它内容取消："
    if ($confirm -ine "YES") {
        Write-Host ""
        Write-Host "已取消操作。"
        Start-Sleep -Seconds 2
        return
    }

    $resolvedTarget = Join-Path $RepoRoot $TargetDir
    if (-not (Test-Path -LiteralPath $resolvedTarget)) {
        Write-Host ""
        Write-Host "[提示] 目标目录不存在，无需清理。"
        Start-Sleep -Seconds 2
        return
    }

    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $resolvedTarget) {
        Write-Host ""
        Write-Host "[错误] 清理失败，请检查文件是否被占用。"
    } else {
        Write-Host ""
        Write-Host "[成功] $TargetDesc 清理完成。"
    }
    Pause-IfNeeded
}

function Show-CleanupMenu {
    while ($true) {
        Clear-Host
        Write-Host "============================================================"
        Write-Host "          系统维护与清理"
        Write-Host "============================================================"
        Write-Host ""
        Write-Host "--- [DDPG+APF 训练产出] ---"
        Write-Host "  [1] 清理 DDPG 模型文件 (multirotor\DDPG_Weight\models)"
        Write-Host "  [2] 清理 DDPG 训练日志 (logs + airsim_training_logs + crazyflie_logs)"
        Write-Host ""
        Write-Host "--- [DQN 控制训练产出] ---"
        Write-Host "  [3] 清理 DQN 模型文件 (multirotor\DQN_Movement\models)"
        Write-Host "  [4] 清理 DQN 训练日志 (logs + scripts\logs)"
        Write-Host ""
        Write-Host "--- [分析结果与全局清理] ---"
        Write-Host "  [5] 清理分析结果 (analysis_results)"
        Write-Host "  [8] 清理全部模型与日志（慎用）"
        Write-Host "  [9] 返回主菜单"
        Write-Host ""
        Write-Host "============================================================"
        Write-Host ""

        $cleanupChoice = Read-Host "请选择维护选项 (1-9)："
        switch -Regex ($cleanupChoice) {
            '^1$' { Confirm-Delete "multirotor\DDPG_Weight\models" "DDPG 权重模型" }
            '^2$' {
                foreach ($target in @(
                    "multirotor\DDPG_Weight\logs",
                    "multirotor\DDPG_Weight\airsim_training_logs",
                    "multirotor\DDPG_Weight\crazyflie_logs"
                )) {
                    if (Test-Path -LiteralPath (Join-Path $RepoRoot $target)) {
                        Remove-Item -LiteralPath (Join-Path $RepoRoot $target) -Recurse -Force -ErrorAction SilentlyContinue
                    }
                }
                Write-Host ""
                Write-Host "[成功] DDPG 日志清理完成。"
                Pause-IfNeeded
            }
            '^3$' { Confirm-Delete "multirotor\DQN_Movement\models" "DQN 模型文件" }
            '^4$' {
                foreach ($target in @(
                    "multirotor\DQN_Movement\logs",
                    "multirotor\DQN_Movement\scripts\logs"
                )) {
                    if (Test-Path -LiteralPath (Join-Path $RepoRoot $target)) {
                        Remove-Item -LiteralPath (Join-Path $RepoRoot $target) -Recurse -Force -ErrorAction SilentlyContinue
                    }
                }
                Write-Host ""
                Write-Host "[成功] DQN 日志清理完成。"
                Pause-IfNeeded
            }
            '^5$' { Confirm-Delete "analysis_results" "分析结果" }
            '^8$' {
                $confirm = Read-Host "请输入 DELETE_ALL 确认执行："
                if ($confirm -ieq "DELETE_ALL") {
                    foreach ($target in @(
                        "multirotor\DDPG_Weight\models",
                        "multirotor\DDPG_Weight\logs",
                        "multirotor\DDPG_Weight\airsim_training_logs",
                        "multirotor\DDPG_Weight\crazyflie_logs",
                        "multirotor\DQN_Movement\models",
                        "multirotor\DQN_Movement\scripts\models",
                        "multirotor\DQN_Movement\logs",
                        "multirotor\DQN_Movement\scripts\logs",
                        "analysis_results"
                    )) {
                        $fullPath = Join-Path $RepoRoot $target
                        if (Test-Path -LiteralPath $fullPath) {
                            Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction SilentlyContinue
                        }
                    }
                    Write-Host ""
                    Write-Host "[成功] 全部训练与分析产出已清理完成。"
                    Pause-IfNeeded
                }
            }
            '^9$' { return }
            default {
                Write-Host ""
                Write-Host "当前输入无效，请重新选择。"
                Start-Sleep -Seconds 2
            }
        }
    }
}

Normalize-BatchFiles

while ($true) {
    Show-ChineseMenu
    $choice = Read-Host "请选择功能选项 (0-9,A-N,R,T,E)："

    switch -Regex ($choice) {
        '^(?i)1$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "系统运行（固定权重）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "run_fixed"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Run_System_Fixed_Weights.bat")
            }
        }
        '^(?i)2$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "系统运行（DDPG 权重预测）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "run_ddpg"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Run_System_DDPG_Weights.bat")
            }
        }
        '^(?i)3$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "系统运行（DQN 控制）"
            Write-Host "============================================================"
            Write-Host ""
            Write-Host "[提示] 该入口暂未开放，建议使用训练入口或评测入口。"
            Write-Host ""
            Pause-IfNeeded
        }
        '^(?i)4$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "DDPG+APF 训练（AirSim，新模型）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "ddpg_train"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Train_DDPG_Weights_Real_Environment.bat")
            }
        }
        '^(?i)5$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "继续 DDPG+APF 训练（AirSim）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "ddpg_train"
            if ($result -ne 0) {
                continue
            }

            $continueModel = Join-Path $RepoRoot "multirotor\DDPG_Weight\models\weight_predictor_airsim.zip"
            if (-not (Test-Path -LiteralPath $continueModel)) {
                Write-Host "[错误] 未检测到可继续训练的 DDPG 模型："
                Write-Host "       $continueModel"
                Write-Host ""
                Pause-IfNeeded
                continue
            }

            [void](Invoke-AsciiBatch "scripts\Train_DDPG_Weights_Real_Environment.bat" @("--continue-model", $continueModel.TrimEnd(".zip")))
        }
        '^(?i)6$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "DDPG+APF 训练（实体日志离线）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "ddpg_logs_train"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Train_DDPG_Weights_Crazyflie_Logs.bat")
            }
        }
        '^(?i)E$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "DDPG+APF 训练（实体无人机单轮在线）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "ddpg_single_episode"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Train_DDPG_Weights_Crazyflie_Online_Single_Episode_EN.bat")
            }
        }
        '^(?i)7$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "DQN 控制训练（AirSim，新模型）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "dqn_train"
            if ($result -eq 0) {
                $previous = $env:USE_PRETRAINED
                $env:USE_PRETRAINED = "0"
                try {
                    [void](Invoke-AsciiBatch "scripts\Train_DQN_Movement_Real_Environment.bat")
                } finally {
                    if ($null -eq $previous) {
                        Remove-Item "Env:USE_PRETRAINED" -ErrorAction SilentlyContinue
                    } else {
                        $env:USE_PRETRAINED = $previous
                    }
                }
            }
        }
        '^(?i)8$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "继续 DQN 控制训练（AirSim）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "dqn_resume_train"
            if ($result -ne 0) {
                continue
            }

            $candidates = @(
                "multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip",
                "multirotor\DQN_Movement\models\movement_dqn_final.zip",
                "multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip",
                "multirotor\DQN_Movement\scripts\models\movement_dqn_final.zip"
            ) | ForEach-Object { Join-Path $RepoRoot $_ }

            if (-not ($candidates | Where-Object { Test-Path -LiteralPath $_ })) {
                Write-Host "[错误] 未检测到可继续训练的 DQN 模型。"
                Write-Host ""
                Pause-IfNeeded
                continue
            }

            $previous = $env:USE_PRETRAINED
            $env:USE_PRETRAINED = "1"
            try {
                [void](Invoke-AsciiBatch "scripts\Train_DQN_Movement_Real_Environment.bat")
            } finally {
                if ($null -eq $previous) {
                    Remove-Item "Env:USE_PRETRAINED" -ErrorAction SilentlyContinue
                } else {
                    $env:USE_PRETRAINED = $previous
                }
            }
        }
        '^(?i)R$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "重新执行当前 stage02 训练"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "dqn_resume_train"
            if ($result -ne 0) {
                continue
            }

            $previousPretrained = $env:USE_PRETRAINED
            $previousStageName = $env:TRAIN_STAGE_NAME
            $previousStageIndex = $env:TRAIN_STAGE_INDEX
            $env:USE_PRETRAINED = "1"
            $env:TRAIN_STAGE_NAME = "stage02_finetune"
            $env:TRAIN_STAGE_INDEX = "2"
            try {
                [void](Invoke-AsciiBatch "scripts\Train_DQN_Movement_Real_Environment.bat")
            } finally {
                foreach ($pair in @(
                    @{ Name = "USE_PRETRAINED"; Value = $previousPretrained },
                    @{ Name = "TRAIN_STAGE_NAME"; Value = $previousStageName },
                    @{ Name = "TRAIN_STAGE_INDEX"; Value = $previousStageIndex }
                )) {
                    if ($null -eq $pair.Value) {
                        Remove-Item ("Env:{0}" -f $pair.Name) -ErrorAction SilentlyContinue
                    } else {
                        Set-Item ("Env:{0}" -f $pair.Name) -Value $pair.Value
                    }
                }
            }
        }
        '^(?i)H$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "分层 DQN 训练（离线 / Mock）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "hrl_train"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Train_Hierarchical_DQN.bat")
            }
        }
        '^(?i)F$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "分层 DQN 训练（AirSim 融合）"
            Write-Host "============================================================"
            Write-Host ""
            $result = Collect-QuickConfig "hrl_train"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Train_Hierarchical_With_AirSim.bat")
            }
        }
        '^(?i)D$' { [void](Invoke-AsciiBatch "scripts\Test_DQN_Movement.bat") }
        '^(?i)A$' { [void](Invoke-AsciiBatch "scripts\Data_Visualization_Analysis.bat") }
        '^(?i)B$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "DDPG 与 DQN 对比分析"
            Write-Host "============================================================"
            Write-Host ""
            $pythonExe = Get-PythonExe
            & $pythonExe (Join-Path $RepoRoot "multirotor\Algorithm\visualize_training_data.py") --compare-algorithms --out (Join-Path $RepoRoot "analysis_results")
            Pause-IfNeeded
        }
        '^(?i)M$' {
            while ($true) {
                Clear-Host
                Write-Host "============================================================"
                Write-Host "四组统一仿真对比阶段"
                Write-Host "============================================================"
                Write-Host ""
                Write-Host "本流程将依次执行以下阶段："
                Write-Host "  [1] APF 基线多轮仿真阶段（fixed APF / random APF）"
                Write-Host "  [2] DDPG+APF stage01 训练"
                Write-Host "  [3] Pure DQN stage01 训练"
                Write-Host "  [4] 在 Unity/AirSim 中执行四组仿真评测（冻结策略）"
                Write-Host "  [5] 训练结果对比分析与 stage02 建议"
                Write-Host ""
                $workflowMode = Resolve-WorkflowMode "comparison"
                if ($workflowMode -eq "__invalid__") {
                    continue
                }
                if ($null -eq $workflowMode) {
                    break
                }

                $result = Collect-QuickConfig "comparison_workflow"
                if ($result -ne 0) {
                    break
                }

                $workflowExit = Run-PaperWorkflow "comparison" $workflowMode
                if ($workflowExit -ne 0) {
                    Write-Host ""
                    Write-Host "[错误] 四组统一仿真对比阶段执行失败，错误码：$workflowExit"
                    Write-Host "[提示] 请检查上方输出中的首个报错信息后再重试。"
                    Write-Host ""
                    Pause-IfNeeded
                }
                if ($env:AIRSIM_TEST_EXIT_AFTER_WORKFLOW -eq "1") {
                    exit 0
                }
                break
            }
        }
        '^(?i)N$' {
            while ($true) {
                Clear-Host
                Write-Host "============================================================"
                Write-Host "虚实两阶段实验工作流 (Virtual-Real Two-Stage Workflow)"
                Write-Host "============================================================"
                Write-Host ""
                $workflowMode = Resolve-WorkflowMode "virtual_real_two_stage"
                if ($workflowMode -eq "__invalid__") {
                    continue
                }
                if ($null -eq $workflowMode) {
                    break
                }

                $result = Collect-QuickConfig "two_stage_workflow"
                if ($result -ne 0) {
                    break
                }

                $workflowExit = Run-PaperWorkflow "virtual_real_two_stage" $workflowMode
                if ($workflowExit -ne 0) {
                    Write-Host ""
                    Write-Host "[错误] 虚实两阶段实验工作流执行失败，错误码：$workflowExit"
                    Write-Host "[提示] 请检查上方输出中的首个报错信息后再重试。"
                    Write-Host ""
                    Pause-IfNeeded
                }
                if ($env:AIRSIM_TEST_EXIT_AFTER_WORKFLOW -eq "1") {
                    exit 0
                }
                break
            }
        }
        '^(?i)G$' {
            $result = Collect-QuickConfig "four_group_benchmark"
            if ($result -eq 0) {
                [void](Invoke-AsciiBatch "scripts\Run_Four_Group_Benchmark.bat")
            }
        }
        '^(?i)I$' { [void](Invoke-AsciiBatch "scripts\Analyze_Four_Group_Benchmark.bat") }
        '^(?i)J$' { [void](Invoke-AsciiBatch "scripts\Analyze_Family_Comparisons.bat") }
        '^(?i)K$' {
            $result = Collect-QuickConfig "paper_ddpg_seeds"
            if ($result -eq 0) {
                & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\Run_Paper_Training_Seeds.ps1") -Algorithm ddpg_apf
            }
        }
        '^(?i)L$' {
            $result = Collect-QuickConfig "paper_dqn_seeds"
            if ($result -eq 0) {
                & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\Run_Paper_Training_Seeds.ps1") -Algorithm pure_dqn
            }
        }
        '^(?i)C$' { Show-CleanupMenu }
        '^(?i)9$' { Show-Info }
        '^(?i)T$' {
            if ($env:AIRSIM_RUNTIME_LOG_MODE -ieq "detail") {
                $env:AIRSIM_RUNTIME_LOG_MODE = "user"
                Write-Host ""
                Write-Host "已切换到用户模式。"
            } else {
                $env:AIRSIM_RUNTIME_LOG_MODE = "detail"
                Write-Host ""
                Write-Host "已切换到详细模式。"
            }
            Write-Host ""
            if ($env:AIRSIM_TEST_EXIT_AFTER_TOGGLE -eq "1") {
                exit 0
            }
            Pause-IfNeeded
        }
        '^(?i)0$' {
            Clear-Host
            Write-Host "============================================================"
            Write-Host "感谢使用 AirSim 无人机仿真平台！"
            Write-Host "============================================================"
            Write-Host ""
            exit 0
        }
        default {
            Write-Host ""
            Write-Host "当前输入无效，请重新选择。"
            Start-Sleep -Seconds 2
        }
    }
}
