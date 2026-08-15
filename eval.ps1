# One-command accuracy eval: regenerate the dataset manifest, ensure the
# static server is up, and open the auto-run eval page in the browser.
#
#   powershell -ExecutionPolicy Bypass -File eval.ps1            # run full set
#   powershell -ExecutionPolicy Bypass -File eval.ps1 -Compare   # sweep configs
param(
  [int]$Port = 4200,
  [switch]$Compare,
  [int]$MaxClips = 0   # 0 = the whole set
)

$here = $PSScriptRoot

# 1. Refresh the manifest so newly added/removed clips are picked up.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here "tools\build-eval-manifest.ps1")

# 2. Start the static server if nothing is listening on $Port.
$up = $false
try { $up = [bool](Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue) } catch {}
if (-not $up) {
  Write-Host "Starting server on $Port..."
  Start-Process powershell -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $here "serve.ps1"), "-Port", "$Port"
  ) -WindowStyle Minimized
  Start-Sleep -Seconds 2
} else {
  Write-Host "Server already running on $Port."
}

# 3. Open the auto-run page.
$run = if ($Compare) { "compare" } else { "vocadito" }
$url = "http://localhost:$Port/eval.html?run=$run"
if ($MaxClips -gt 0) { $url += "&max=$MaxClips" }
Write-Host "Opening $url"
Start-Process $url
