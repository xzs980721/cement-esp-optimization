param(
    [string]$TectonicPath = ""
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$paperDir = Join-Path $projectRoot "paper"
$buildDir = Join-Path $projectRoot "build\latex"
$finalDir = Join-Path $projectRoot "output\pdf"
$finalPdf = Join-Path $finalDir "Cement_ESP_Optimization_Paper.pdf"

if ([string]::IsNullOrWhiteSpace($TectonicPath)) {
    $command = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        $TectonicPath = $command.Source
    } else {
        $portable = Join-Path ([System.IO.Path]::GetTempPath()) "codex_tectonic_0.16.9\tectonic.exe"
        if (Test-Path -LiteralPath $portable) {
            $TectonicPath = $portable
        }
    }
}

if (-not (Test-Path -LiteralPath $TectonicPath)) {
    throw "Tectonic was not found. Install it from https://tectonic-typesetting.github.io/ or pass -TectonicPath."
}

New-Item -ItemType Directory -Force -Path $buildDir, $finalDir | Out-Null
Push-Location $paperDir
try {
    & $TectonicPath -X compile main.tex --outdir $buildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX compilation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Copy-Item -LiteralPath (Join-Path $buildDir "main.pdf") -Destination $finalPdf -Force
Write-Output $finalPdf
