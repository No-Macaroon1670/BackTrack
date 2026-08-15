# Minimal static file server for the backtrack app (no Node/Python required).
#
# Built on a raw TcpListener rather than HttpListener deliberately:
# HttpListener registers URL prefixes with the http.sys kernel driver, and
# when the process is hard-killed that registration can be orphaned, landing
# the port in Windows' excluded port list (netsh int ipv4 show
# excludedportrange). A plain socket's worst case is ~2 minutes of TIME_WAIT.
param([int]$Port = 4200)

$root = $PSScriptRoot
$mime = @{
  ".html" = "text/html; charset=utf-8"
  ".css"  = "text/css; charset=utf-8"
  ".js"   = "text/javascript; charset=utf-8"
  ".json" = "application/json"
  ".wav"  = "audio/wav"
  ".mid"  = "audio/midi"
  ".svg"  = "image/svg+xml"
  ".png"  = "image/png"
  ".ico"  = "image/x-icon"
}

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
$listener.Start()
Write-Host "Serving $root at http://localhost:$Port/"

function Send-Response($stream, [int]$code, [string]$status, [string]$type, [byte[]]$body) {
  $header = "HTTP/1.1 $code $status`r`n" +
            "Content-Type: $type`r`n" +
            "Content-Length: $($body.Length)`r`n" +
            "Cache-Control: no-store`r`n" +
            "Connection: close`r`n`r`n"
  $hb = [System.Text.Encoding]::ASCII.GetBytes($header)
  $stream.Write($hb, 0, $hb.Length)
  if ($body.Length -gt 0) { $stream.Write($body, 0, $body.Length) }
  $stream.Flush()
}

try {
  while ($true) {
    $client = $listener.AcceptTcpClient()
    try {
      $stream = $client.GetStream()
      $stream.ReadTimeout = 5000
      $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::ASCII)
      $requestLine = $reader.ReadLine()
      while ($true) {
        $line = $reader.ReadLine()
        if ($null -eq $line -or $line -eq "") { break }
      }

      if ($requestLine -match '^GET\s+([^\s\?#]+)') {
        $rel = [System.Uri]::UnescapeDataString($Matches[1]).TrimStart("/")
        if ($rel -eq "") { $rel = "index.html" }
        $full = [System.IO.Path]::GetFullPath((Join-Path $root $rel))
        if ($full.StartsWith($root) -and (Test-Path $full -PathType Leaf)) {
          $ext = [System.IO.Path]::GetExtension($full).ToLower()
          $type = $mime[$ext]
          if ($null -eq $type) { $type = "application/octet-stream" }
          Send-Response $stream 200 "OK" $type ([System.IO.File]::ReadAllBytes($full))
        } else {
          Send-Response $stream 404 "Not Found" "text/plain" ([System.Text.Encoding]::UTF8.GetBytes("404 Not Found"))
        }
      } else {
        Send-Response $stream 405 "Method Not Allowed" "text/plain" ([System.Text.Encoding]::UTF8.GetBytes("GET only"))
      }
    } catch {
      # per-connection errors (timeouts, resets) shouldn't kill the server
    } finally {
      $client.Close()
    }
  }
} finally {
  $listener.Stop()
}
