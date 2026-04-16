$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $root 'analysis_results'
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$outputPath = Join-Path $outputDir 'DDPG_APF_Logic_Flowchart.png'

$width = 1600
$height = 2300
$bmp = New-Object System.Drawing.Bitmap $width, $height
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

function Draw-RoundedBox {
    param(
        [int]$X,
        [int]$Y,
        [int]$W,
        [int]$H,
        [string]$Text
    )

    $radius = 18
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc($X, $Y, $radius, $radius, 180, 90)
    $path.AddArc($X + $W - $radius, $Y, $radius, $radius, 270, 90)
    $path.AddArc($X + $W - $radius, $Y + $H - $radius, $radius, $radius, 0, 90)
    $path.AddArc($X, $Y + $H - $radius, $radius, $radius, 90, 90)
    $path.CloseFigure()
    $g.FillPath($boxBrush, $path)
    $g.DrawPath($pen, $path)
    $rect = New-Object System.Drawing.RectangleF ($X + 10), ($Y + 8), ($W - 20), ($H - 16)
    $g.DrawString($Text, $boxFont, $textBrush, $rect, $centerFormat)
    $path.Dispose()
}

function Draw-Box {
    param(
        [int]$X,
        [int]$Y,
        [int]$W,
        [int]$H,
        [string]$Text
    )

    $rect = New-Object System.Drawing.Rectangle $X, $Y, $W, $H
    $g.FillRectangle($boxBrush, $rect)
    $g.DrawRectangle($pen, $rect)
    $textRect = New-Object System.Drawing.RectangleF ($X + 12), ($Y + 10), ($W - 24), ($H - 20)
    $g.DrawString($Text, $boxFont, $textBrush, $textRect, $centerFormat)
}

function Draw-Diamond {
    param(
        [int]$CenterX,
        [int]$Y,
        [int]$W,
        [int]$H,
        [string]$Text
    )

    $points = [System.Drawing.Point[]]@(
        (New-Object System.Drawing.Point $CenterX, $Y),
        (New-Object System.Drawing.Point ($CenterX + [int]($W / 2)), ($Y + [int]($H / 2))),
        (New-Object System.Drawing.Point $CenterX, ($Y + $H)),
        (New-Object System.Drawing.Point ($CenterX - [int]($W / 2)), ($Y + [int]($H / 2)))
    )
    $g.FillPolygon($decisionBrush, $points)
    $g.DrawPolygon($pen, $points)
    $rect = New-Object System.Drawing.RectangleF ($CenterX - [int]($W / 2) + 18), ($Y + 18), ($W - 36), ($H - 36)
    $g.DrawString($Text, $boxFont, $textBrush, $rect, $centerFormat)
}

function Draw-Arrow {
    param(
        [int]$X1,
        [int]$Y1,
        [int]$X2,
        [int]$Y2,
        [string]$Label = ''
    )

    $g.DrawLine($arrowPen, $X1, $Y1, $X2, $Y2)
    if ($Label) {
        $labelRect = New-Object System.Drawing.RectangleF ([Math]::Min($X1, $X2) + 8), ([Math]::Min($Y1, $Y2) - 18), 80, 24
        $g.DrawString($Label, $smallFont, $textBrush, $labelRect, $leftFormat)
    }
}

function Draw-PolylineArrow {
    param(
        [int[]]$Points,
        [string]$Label = ''
    )

    for ($i = 0; $i -lt $Points.Length - 2; $i += 2) {
        $x1 = $Points[$i]
        $y1 = $Points[$i + 1]
        $x2 = $Points[$i + 2]
        $y2 = $Points[$i + 3]
        if ($i -eq $Points.Length - 4) {
            $g.DrawLine($arrowPen, $x1, $y1, $x2, $y2)
        } else {
            $g.DrawLine($pen, $x1, $y1, $x2, $y2)
        }
    }
    if ($Label) {
        $labelRect = New-Object System.Drawing.RectangleF ($Points[0] + 10), ($Points[1] - 22), 60, 24
        $g.DrawString($Label, $smallFont, $textBrush, $labelRect, $leftFormat)
    }
}

$titleRect = New-Object System.Drawing.RectangleF 0, 24, $width, 36
$g.DrawString('DDPG+APF 逻辑流程图', $titleFont, $textBrush, $titleRect, $centerFormat)
$g.DrawLine($pen, 90, 82, ($width - 90), 82)

$centerX = 800
$startX = 570
$mainX = 460
$mainW = 680

Draw-RoundedBox $startX 118 460 78 '开始'
Draw-Box $mainX 232 $mainW 104 '初始化网络参数、目标网络、经验池 D、探索噪声 ψ，以及 Server / Env'
Draw-Box $mainX 370 $mainW 104 '重置环境，获取初始状态 s_t，并初始化当前 episode 的电量、碰撞和扫描状态'
Draw-Box $mainX 508 $mainW 110 'Actor 根据状态 s_t 输出动作 a_t，并加入探索噪声得到执行动作 ã_t'
Draw-Box $mainX 652 $mainW 110 '对 ã_t 做裁剪、平滑和安全约束，并映射为 APF 权重与避障参数'
Draw-Box $mainX 796 $mainW 118 '计算 F_repulsion、F_entropy、F_distance、F_leader、F_history、F_obs'
Draw-Box $mainX 950 $mainW 118 '融合得到最终方向 F_t，执行飞行，获取奖励 r_t、下一状态 s_{t+1} 和 done'
Draw-Box $mainX 1104 $mainW 112 '将经验存入 D，采样批数据，更新 Critic / Actor，并软更新目标网络'
Draw-Diamond $centerX 1260 320 136 'done = True ?'
Draw-Box $mainX 1446 $mainW 104 '保存最优模型 θ_μ* 与最优 APF 参数集合 W*'
Draw-Box $mainX 1584 $mainW 110 '进入评测阶段：加载最优模型，按 μ(s_t|θ_μ*) 生成 APF 参数并执行扫描'
Draw-Box $mainX 1728 $mainW 110 '记录扫描率、平均熵、碰撞次数、电量和轨迹数据'
Draw-Diamond $centerX 1878 320 136 '达到扫描目标或触发终止条件 ?'
Draw-RoundedBox $startX 2066 460 80 '输出日志、统计结果和对比图表'

Draw-Arrow $centerX 196 $centerX 232
Draw-Arrow $centerX 336 $centerX 370
Draw-Arrow $centerX 474 $centerX 508
Draw-Arrow $centerX 618 $centerX 652
Draw-Arrow $centerX 762 $centerX 796
Draw-Arrow $centerX 914 $centerX 950
Draw-Arrow $centerX 1068 $centerX 1104
Draw-Arrow $centerX 1216 $centerX 1260
Draw-Arrow $centerX 1396 $centerX 1446 '是'
Draw-Arrow $centerX 1550 $centerX 1584
Draw-Arrow $centerX 1694 $centerX 1728
Draw-Arrow $centerX 1838 $centerX 1878
Draw-Arrow $centerX 2014 $centerX 2066 '是'

Draw-PolylineArrow @(960, 1328, 1300, 1328, 1300, 562, 1140, 562) '否'
Draw-PolylineArrow @(960, 1946, 1300, 1946, 1300, 1640, 1140, 1640) '否'

$captionRect = New-Object System.Drawing.RectangleF 140, 2176, 1320, 54
$g.DrawString('图示说明：上半部分为 DDPG+APF 学习训练流程，下半部分为冻结策略后的输出与评测流程。', $smallFont, $textBrush, $captionRect, $centerFormat)

$bmp.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$titleFont.Dispose()
$boxFont.Dispose()
$smallFont.Dispose()
$pen.Dispose()
$arrowPen.Dispose()
$boxBrush.Dispose()
$decisionBrush.Dispose()
$g.Dispose()
$bmp.Dispose()

Write-Output $outputPath
