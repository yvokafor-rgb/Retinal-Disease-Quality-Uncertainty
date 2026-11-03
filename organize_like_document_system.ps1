# Run from the root of the already-cloned Retinal-Disease-Quality-Uncertainty repository.
# Default PREVIEW mode; use -Apply only after reviewing.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.git')) { throw 'Open PowerShell in the cloned repository root.' }
$root = (Get-Location).Path
function Ensure-Dir($path) { if ($Apply) { New-Item -ItemType Directory -Force -Path $path | Out-Null } }
function Safe-Move($source, $dest) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { return }
    if (Test-Path -LiteralPath $dest) { Write-Warning "Already exists, skipped: $dest"; return }
    Write-Host "$source -> $dest"
    if ($Apply) { Ensure-Dir (Split-Path $dest -Parent); Move-Item -LiteralPath $source -Destination $dest }
}
# No dataset deletion or movement; leave EyeQ and RFMiD untouched.
foreach ($f in @('preprocessing.py','model_training.py','eyeq_quality.py','model_deployment.py')) {
    Safe-Move $f (Join-Path 'src' $f)
}
# Do NOT flatten results: preserve the full original RFMiD_Results hierarchy.
if (Test-Path -LiteralPath 'RFMiD_Results') {
    if (Test-Path -LiteralPath 'results/RFMiD_Results') { Write-Warning 'results/RFMiD_Results already exists; skipped.' }
    else {
        Write-Host 'RFMiD_Results -> results/RFMiD_Results (preserve original hierarchy)'
        if ($Apply) { Ensure-Dir 'results'; Move-Item -LiteralPath 'RFMiD_Results' -Destination 'results/RFMiD_Results' }
    }
}
Safe-Move 'RFMiD_Research_Submission.zip' 'docs/RFMiD_Research_Submission.zip'
# Do not alter src/ contents, overwrite README, or stage/commit changes automatically.
Write-Host "`nReview README, import paths and dataset paths before executing moved scripts."
if (-not $Apply) { Write-Host 'Preview only. To apply: .\organize_like_document_system.ps1 -Apply' }
else { Write-Host 'Applied. Next: git status --short; test scripts; review git diff before commit.' }
