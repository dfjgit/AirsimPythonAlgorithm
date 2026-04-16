[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$MainBatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Normalize-CrlfBytes {
    param(
        [byte[]]$Bytes
    )

    $latin1 = [System.Text.Encoding]::GetEncoding(28591)
    $text = $latin1.GetString($Bytes)
    $normalized = $text.Replace("`r`n", "`n").Replace("`r", "`n").Replace("`n", "`r`n")
    return $latin1.GetBytes($normalized)
}

$rootPath = [System.IO.Path]::GetFullPath($RepoRoot)
if ([System.IO.Path]::IsPathRooted($MainBatch)) {
    $mainBatchPath = [System.IO.Path]::GetFullPath($MainBatch)
} else {
    $mainBatchPath = [System.IO.Path]::GetFullPath((Join-Path $rootPath $MainBatch))
}

$batchFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
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

if (-not (Test-Path -LiteralPath $mainBatchPath)) {
    Write-Error "Main batch file not found: $mainBatchPath"
    exit 1
}

& $mainBatchPath
exit $LASTEXITCODE
