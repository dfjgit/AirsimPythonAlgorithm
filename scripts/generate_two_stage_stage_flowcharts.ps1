$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $root 'analysis_results'
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$stage1Path = Join-Path $outputDir 'TwoStage_Stage1_Flowchart.png'
$stage2Path = Join-Path $outputDir 'TwoStage_Stage2_Flowchart.png'

function New-DrawingContext {
    param(
        [int]$Width,
        [int]$Height
    )

    $bmp = New-Object System.Drawing.Bitmap $Width, $Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)

    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 2
    $arrowCap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap 6, 8, $true
    $arrowPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 2
    $arrowPen.CustomEndCap = $arrowCap
    $boxBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(248, 248, 248))
    $decisionBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(242, 242, 242))
    $textBrush = [System.Drawing.Brushes]::Black

    $titleFont = New-Object System.Drawing.Font ('Microsoft YaHei', 20, [System.Drawing.FontStyle]::Bold)
    $boxFont = New-Object System.Drawing.Font ('Microsoft YaHei', 13, [System.Drawing.FontStyle]::Regular)
    $smallFont = New-Object System.Drawing.Font ('Microsoft YaHei', 12, [System.Drawing.FontStyle]::Regular)

    $centerFormat = New-Object System.Drawing.StringFormat
    $centerFormat.Alignment = [System.Drawing.StringAlignment]::Center
    $centerFormat.LineAlignment = [System.Drawing.StringAlignment]::Center
    $centerFormat.FormatFlags = [System.Drawing.StringFormatFlags]::LineLimit

    $leftFormat = New-Object System.Drawing.StringFormat
    $leftFormat.Alignment = [System.Drawing.StringAlignment]::Near
    $leftFormat.LineAlignment = [System.Drawing.StringAlignment]::Center

    return @{
        Bitmap = $bmp
        Graphics = $g
        Pen = $pen
        ArrowPen = $arrowPen
        BoxBrush = $boxBrush
        DecisionBrush = $decisionBrush
        TextBrush = $textBrush
        TitleFont = $titleFont
        BoxFont = $boxFont
        SmallFont = $smallFont
        CenterFormat = $centerFormat
        LeftFormat = $leftFormat
    }
}

function Close-DrawingContext {
    param($ctx)
    $ctx.TitleFont.Dispose()
    $ctx.BoxFont.Dispose()
    $ctx.SmallFont.Dispose()
    $ctx.Pen.Dispose()
    $ctx.ArrowPen.Dispose()
    $ctx.BoxBrush.Dispose()
    $ctx.DecisionBrush.Dispose()
    $ctx.Graphics.Dispose()
    $ctx.Bitmap.Dispose()
}

function Draw-RoundedBox {
    param($ctx, [int]$X, [int]$Y, [int]$W, [int]$H, [string]$Text)
    $radius = 18
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc($X, $Y, $radius, $radius, 180, 90)
    $path.AddArc($X + $W - $radius, $Y, $radius, $radius, 270, 90)
    $path.AddArc($X + $W - $radius, $Y + $H - $radius, $radius, $radius, 0, 90)
    $path.AddArc($X, $Y + $H - $radius, $radius, $radius, 90, 90)
    $path.CloseFigure()
    $ctx.Graphics.FillPath($ctx.BoxBrush, $path)
    $ctx.Graphics.DrawPath($ctx.Pen, $path)
    $rect = New-Object System.Drawing.RectangleF ($X + 10), ($Y + 8), ($W - 20), ($H - 16)
    $ctx.Graphics.DrawString($Text, $ctx.BoxFont, $ctx.TextBrush, $rect, $ctx.CenterFormat)
    $path.Dispose()
}

function Draw-Box {
    param($ctx, [int]$X, [int]$Y, [int]$W, [int]$H, [string]$Text)
    $rect = New-Object System.Drawing.Rectangle $X, $Y, $W, $H
    $ctx.Graphics.FillRectangle($ctx.BoxBrush, $rect)
    $ctx.Graphics.DrawRectangle($ctx.Pen, $rect)
    $textRect = New-Object System.Drawing.RectangleF ($X + 12), ($Y + 10), ($W - 24), ($H - 20)
    $ctx.Graphics.DrawString($Text, $ctx.BoxFont, $ctx.TextBrush, $textRect, $ctx.CenterFormat)
}

function Draw-Diamond {
    param($ctx, [int]$CenterX, [int]$Y, [int]$W, [int]$H, [string]$Text)
    $points = [System.Drawing.Point[]]@(
        (New-Object System.Drawing.Point $CenterX, $Y),
        (New-Object System.Drawing.Point ($CenterX + [int]($W / 2)), ($Y + [int]($H / 2))),
        (New-Object System.Drawing.Point $CenterX, ($Y + $H)),
        (New-Object System.Drawing.Point ($CenterX - [int]($W / 2)), ($Y + [int]($H / 2)))
    )
    $ctx.Graphics.FillPolygon($ctx.DecisionBrush, $points)
    $ctx.Graphics.DrawPolygon($ctx.Pen, $points)
    $rect = New-Object System.Drawing.RectangleF ($CenterX - [int]($W / 2) + 18), ($Y + 18), ($W - 36), ($H - 36)
    $ctx.Graphics.DrawString($Text, $ctx.BoxFont, $ctx.TextBrush, $rect, $ctx.CenterFormat)
}

function Draw-Arrow {
    param($ctx, [int]$X1, [int]$Y1, [int]$X2, [int]$Y2, [string]$Label = '')
    $ctx.Graphics.DrawLine($ctx.ArrowPen, $X1, $Y1, $X2, $Y2)
    if ($Label) {
        $labelRect = New-Object System.Drawing.RectangleF ([Math]::Min($X1, $X2) + 8), ([Math]::Min($Y1, $Y2) - 18), 90, 24
        $ctx.Graphics.DrawString($Label, $ctx.SmallFont, $ctx.TextBrush, $labelRect, $ctx.LeftFormat)
    }
}

function Draw-PolylineArrow {
    param($ctx, [int[]]$Points, [string]$Label = '')
    for ($i = 0; $i -lt $Points.Length - 2; $i += 2) {
        $x1 = $Points[$i]
        $y1 = $Points[$i + 1]
        $x2 = $Points[$i + 2]
        $y2 = $Points[$i + 3]
        if ($i -eq $Points.Length - 4) {
            $ctx.Graphics.DrawLine($ctx.ArrowPen, $x1, $y1, $x2, $y2)
        } else {
            $ctx.Graphics.DrawLine($ctx.Pen, $x1, $y1, $x2, $y2)
        }
    }
    if ($Label) {
        $labelRect = New-Object System.Drawing.RectangleF ($Points[0] + 8), ($Points[1] - 22), 60, 24
        $ctx.Graphics.DrawString($Label, $ctx.SmallFont, $ctx.TextBrush, $labelRect, $ctx.LeftFormat)
    }
}

function Draw-Header {
    param($ctx, [int]$Width, [string]$Title)
    $titleRect = New-Object System.Drawing.RectangleF 0, 24, $Width, 38
    $ctx.Graphics.DrawString($Title, $ctx.TitleFont, $ctx.TextBrush, $titleRect, $ctx.CenterFormat)
    $ctx.Graphics.DrawLine($ctx.Pen, 90, 82, ($Width - 90), 82)
}

function Save-Stage1Flowchart {
    param([string]$Path)
    $ctx = New-DrawingContext -Width 1600 -Height 2200
    try {
        Draw-Header $ctx 1600 '双阶段实验：阶段一（sim_pretrain）流程图'
        $cx = 800
        $sx = 560
        $mx = 430
        $mw = 740

        Draw-RoundedBox $ctx $sx 120 480 78 '开始'
        Draw-Box $ctx $mx 240 $mw 108 '读取系统配置、训练配置和奖励配置'
        Draw-Box $ctx $mx 384 $mw 108 '初始化 DDPG 网络参数、目标网络、经验池 D、探索噪声 ψ'
        Draw-Box $ctx $mx 528 $mw 108 '启动 Unity / AirSim，初始化多无人机算法服务器 Server 和训练环境 Env'
        Draw-Box $ctx $mx 672 $mw 108 '重置环境，获取初始状态 s_t，并初始化当前 episode 的统计量'
        Draw-Box $ctx $mx 816 $mw 112 'Actor 根据 s_t 输出动作 a_t，加入噪声并映射为 APF 权重与避障参数'
        Draw-Box $ctx $mx 966 $mw 118 '计算 F_repulsion、F_entropy、F_distance、F_leader、F_history、F_obs，并融合得到 F_t'
        Draw-Box $ctx $mx 1124 $mw 112 '在 AirSim 中执行动作，获取 r_t、s_{t+1}、done，并写入经验池 D'
        Draw-Box $ctx $mx 1270 $mw 110 '从 D 中采样批数据，更新 Critic / Actor，并软更新目标网络'
        Draw-Diamond $ctx $cx 1428 320 136 'done = True ?'
        Draw-Box $ctx $mx 1612 $mw 104 '若当前 episode 结束，则重置环境并进入下一轮训练'
        Draw-Diamond $ctx $cx 1760 320 136 '达到阶段一训练预算 ?'
        Draw-Box $ctx $mx 1948 $mw 104 '保存仿真预训练模型 weight_predictor_airsim 与阶段一训练日志'
        Draw-RoundedBox $ctx $sx 2088 480 78 '输出阶段一结果'

        Draw-Arrow $ctx $cx 198 $cx 240
        Draw-Arrow $ctx $cx 348 $cx 384
        Draw-Arrow $ctx $cx 492 $cx 528
        Draw-Arrow $ctx $cx 636 $cx 672
        Draw-Arrow $ctx $cx 780 $cx 816
        Draw-Arrow $ctx $cx 928 $cx 966
        Draw-Arrow $ctx $cx 1084 $cx 1124
        Draw-Arrow $ctx $cx 1236 $cx 1270
        Draw-Arrow $ctx $cx 1380 $cx 1428
        Draw-Arrow $ctx $cx 1564 $cx 1612 '是'
        Draw-Arrow $ctx $cx 1716 $cx 1760
        Draw-Arrow $ctx $cx 1896 $cx 1948 '是'
        Draw-Arrow $ctx $cx 2052 $cx 2088

        Draw-PolylineArrow $ctx @(960, 1496, 1290, 1496, 1290, 726, 1170, 726) '否'
        Draw-PolylineArrow $ctx @(640, 1828, 300, 1828, 300, 726, 430, 726) '否'

        $captionRect = New-Object System.Drawing.RectangleF 140, 2170, 1320, 24
        $ctx.Graphics.DrawString('图示说明：阶段一在 Unity/AirSim 中完成 DDPG+APF 的仿真预训练，并输出预训练模型与日志。', $ctx.SmallFont, $ctx.TextBrush, $captionRect, $ctx.CenterFormat)

        $ctx.Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        Close-DrawingContext $ctx
    }
}

function Save-Stage2Flowchart {
    param([string]$Path)
    $ctx = New-DrawingContext -Width 1700 -Height 2450
    try {
        Draw-Header $ctx 1700 '双阶段实验：阶段二（real_weighted_refine）流程图'
        $cx = 850
        $sx = 610
        $mx = 310
        $mw = 1080
        $leftX = 90
        $rightX = 1090
        $branchW = 520

        Draw-RoundedBox $ctx $sx 120 480 78 '开始'
        Draw-Box $ctx $mx 238 $mw 104 '读取阶段一输出模型 M_sim；若继续修正，则读取当前阶段二模型作为新的初始模型'
        Draw-Diamond $ctx $cx 390 360 150 'refine_mode = online ?'

        Draw-Box $ctx $leftX 620 $branchW 112 '在线分支：连接实体 Crazyflie，解析默认模型路径，并构建 CrazyflieOnlineWeightEnv'
        Draw-Box $ctx $leftX 780 $branchW 112 '执行单轮或加权在线修正，按真实飞行回合更新 DDPG 权重，并保存 online 模型与日志'

        Draw-Box $ctx $rightX 620 $branchW 112 '离线分支：读取 Crazyflie 历史日志 L_real，并构建 CrazyflieLogEnv'
        Draw-Box $ctx $rightX 780 $branchW 112 '继续训练阶段一模型，利用日志完成离线修正，并保存 offline_logs 模型与日志'

        Draw-Box $ctx $mx 1020 $mw 104 '归档阶段二模型、日志和阶段元数据到 workflow artifacts/real_weighted_refine/<mode>'
        Draw-Box $ctx $mx 1160 $mw 104 '读取阶段一与阶段二训练 CSV，构建 two_stage_summary.csv 与 two_stage_summary.md'
        Draw-Box $ctx $mx 1300 $mw 110 '计算 efficiency gain、success gain，并执行继续 / 谨慎继续 / 停止 的推荐判断'
        Draw-Diamond $ctx $cx 1458 360 150 '是否继续实飞修正 ?'

        Draw-Box $ctx 90 1710 560 108 '继续：保留当前修正模型并进入下一轮 real_weighted_refine'
        Draw-Box $ctx 1050 1710 560 108 '停止：锁定当前修正模型、总结分析结果并结束双阶段实验'
        Draw-RoundedBox $ctx $sx 1980 480 78 '输出阶段二结果'

        Draw-Arrow $ctx $cx 198 $cx 238
        Draw-Arrow $ctx $cx 342 $cx 390

        Draw-PolylineArrow $ctx @(670, 465, 350, 465, 350, 620)
        Draw-PolylineArrow $ctx @(1030, 465, 1350, 465, 1350, 620)
        $ctx.Graphics.DrawString('是', $ctx.SmallFont, $ctx.TextBrush, (New-Object System.Drawing.RectangleF 560, 442, 40, 24), $ctx.LeftFormat)
        $ctx.Graphics.DrawString('否', $ctx.SmallFont, $ctx.TextBrush, (New-Object System.Drawing.RectangleF 1040, 442, 40, 24), $ctx.LeftFormat)

        Draw-Arrow $ctx 350 732 350 780
        Draw-Arrow $ctx 1350 732 1350 780

        Draw-PolylineArrow $ctx @(350, 892, 350, 960, 620, 960, 620, 1020)
        Draw-PolylineArrow $ctx @(1350, 892, 1350, 960, 1080, 960, 1080, 1020)

        Draw-Arrow $ctx $cx 1124 $cx 1160
        Draw-Arrow $ctx $cx 1264 $cx 1300
        Draw-Arrow $ctx $cx 1410 $cx 1458

        Draw-PolylineArrow $ctx @(670, 1533, 370, 1533, 370, 1710)
        Draw-PolylineArrow $ctx @(1030, 1533, 1330, 1533, 1330, 1710)
        $ctx.Graphics.DrawString('是', $ctx.SmallFont, $ctx.TextBrush, (New-Object System.Drawing.RectangleF 560, 1510, 40, 24), $ctx.LeftFormat)
        $ctx.Graphics.DrawString('否', $ctx.SmallFont, $ctx.TextBrush, (New-Object System.Drawing.RectangleF 1040, 1510, 40, 24), $ctx.LeftFormat)

        Draw-PolylineArrow $ctx @(370, 1818, 370, 1910, 850, 1910, 850, 1980)
        Draw-PolylineArrow $ctx @(1330, 1818, 1330, 1910, 850, 1910)

        Draw-PolylineArrow $ctx @(90, 1764, 40, 1764, 40, 290, 310, 290)
        $ctx.Graphics.DrawString('循环', $ctx.SmallFont, $ctx.TextBrush, (New-Object System.Drawing.RectangleF 52, 1738, 60, 24), $ctx.LeftFormat)

        $captionRect = New-Object System.Drawing.RectangleF 150, 2080, 1400, 64
        $ctx.Graphics.DrawString('图示说明：阶段二以阶段一模型为起点，根据 refine_mode 进入在线修正或离线日志修正分支，最后统一生成双阶段汇总与继续建议。', $ctx.SmallFont, $ctx.TextBrush, $captionRect, $ctx.CenterFormat)

        $ctx.Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        Close-DrawingContext $ctx
    }
}

Save-Stage1Flowchart -Path $stage1Path
Save-Stage2Flowchart -Path $stage2Path

Write-Output $stage1Path
Write-Output $stage2Path
