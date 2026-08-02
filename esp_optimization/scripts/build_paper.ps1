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
        } else {
            $bundled = Join-Path $env:USERPROFILE ".codex\.tmp\bundled-marketplaces\openai-bundled\plugins\latex\bin\tectonic.exe"
            if (Test-Path -LiteralPath $bundled) {
                $TectonicPath = $bundled
            }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($TectonicPath) -or -not (Test-Path -LiteralPath $TectonicPath)) {
    throw "Tectonic was not found. Install it from https://tectonic-typesetting.github.io/ or pass -TectonicPath."
}

New-Item -ItemType Directory -Force -Path $buildDir, $finalDir | Out-Null
$builtPdf = Join-Path $buildDir "main.pdf"
if (Test-Path -LiteralPath $builtPdf) {
    Remove-Item -LiteralPath $builtPdf -Force
}
Push-Location $paperDir
try {
    & $TectonicPath -X compile main.tex --outdir $buildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX compilation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $builtPdf)) {
    throw "LaTeX compilation did not produce main.pdf."
}
Copy-Item -LiteralPath $builtPdf -Destination $finalPdf -Force
Write-Output $finalPdf
