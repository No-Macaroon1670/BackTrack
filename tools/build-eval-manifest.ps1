# Scans the Vocadito dataset folder and writes vocadito/manifest.json so
# eval.html can auto-load every clip over HTTP (the browser can't list a
# directory). Run whenever the dataset contents change.
#
#   powershell -ExecutionPolicy Bypass -File tools/build-eval-manifest.ps1
param(
  [string]$Dataset = (Join-Path $PSScriptRoot "..\vocadito")
)

$root = (Resolve-Path $Dataset).Path
$audioDir = Join-Path $root "Audio"
if (-not (Test-Path $audioDir)) { Write-Error "No Audio/ under $root"; exit 1 }

$clips = @()
Get-ChildItem (Join-Path $audioDir "*.wav") |
  Where-Object { $_.Name -notmatch '^\._' -and $_.FullName -notmatch '__MACOSX' } |
  Sort-Object { [int]($_.BaseName -replace '\D', '') } |
  ForEach-Object {
    $id = $_.BaseName                       # e.g. vocadito_10
    $rel = "vocadito"                        # URL prefix from the served root
    $entry = [ordered]@{ id = $id; audio = "$rel/Audio/$($_.Name)" }

    $f0 = Join-Path $root "Annotations\F0\${id}_f0.csv"
    if (Test-Path $f0) { $entry.f0 = "$rel/Annotations/F0/${id}_f0.csv" }

    foreach ($ann in @("A1", "A2")) {
      $notes = Join-Path $root "Annotations\Notes\${id}_notes$ann.csv"
      if (Test-Path $notes) { $entry["notes$ann"] = "$rel/Annotations/Notes/${id}_notes$ann.csv" }
    }
    $clips += $entry
  }

$manifest = [ordered]@{
  dataset = "vocadito"
  generated = (Get-Date).ToString("s")
  noteFormat = "vocadito"   # start_sec, pitch_Hz, duration_sec
  count = $clips.Count
  clips = $clips
}

$out = Join-Path $root "manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Out-File -FilePath $out -Encoding utf8
Write-Host "Wrote $out with $($clips.Count) clips."
