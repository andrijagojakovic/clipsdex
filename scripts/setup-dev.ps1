<#
.SYNOPSIS
  Checks (and optionally installs) the developer tools ClipsDex needs on Windows.

.DESCRIPTION
  Run without arguments to only CHECK what is installed:
      powershell -ExecutionPolicy Bypass -File scripts\setup-dev.ps1

  Run with -Install to install whatever is missing with winget
  (Windows will ask for permission for some installers):
      powershell -ExecutionPolicy Bypass -File scripts\setup-dev.ps1 -Install

  After installing, close and reopen the terminal so the new PATH is picked up,
  then run the script again without -Install to confirm everything is found.

  This is a developer script. End users never run it; the desktop app's
  first-run wizard will handle their setup (Phase 8).
#>
param(
    [switch]$Install,
    [string]$OllamaModel = "gemma3:4b"
)

$ErrorActionPreference = "Stop"

# Each tool: display name, command used to detect it, winget package id, extra winget args.
$tools = @(
    @{ Name = "Git";                Cmd = "git";      Id = "Git.Git" },
    @{ Name = "uv (Python manager)"; Cmd = "uv";      Id = "astral-sh.uv" },
    @{ Name = "Node.js LTS";        Cmd = "node";     Id = "OpenJS.NodeJS.LTS" },
    @{ Name = "pnpm";               Cmd = "pnpm";     Id = "pnpm.pnpm" },
    @{ Name = "Rust (rustup)";      Cmd = "rustup";   Id = "Rustlang.Rustup" },
    @{ Name = "ffmpeg + ffprobe";   Cmd = "ffmpeg";   Id = "Gyan.FFmpeg" },
    @{ Name = "Chromaprint fpcalc"; Cmd = "fpcalc";   Id = "AcoustID.Chromaprint" },
    @{ Name = "Ollama";             Cmd = "ollama";   Id = "Ollama.Ollama" }
)

function Test-Command([string]$cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Test-VCBuildTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return [bool]$found
}

function Install-WithWinget([string]$id, [string[]]$extra = @()) {
    $args = @("install", "--id", $id, "-e", "--accept-package-agreements", "--accept-source-agreements") + $extra
    Write-Host "    winget $($args -join ' ')" -ForegroundColor DarkGray
    & winget @args
}

if ($Install -and -not (Test-Command "winget")) {
    Write-Host "winget is not available. Install 'App Installer' from the Microsoft Store, then run this script again." -ForegroundColor Red
    exit 1
}

$missing = @()
Write-Host ""
Write-Host "ClipsDex developer setup check" -ForegroundColor Cyan
Write-Host "-------------------------------"

foreach ($t in $tools) {
    if (Test-Command $t.Cmd) {
        Write-Host ("[OK]      {0}" -f $t.Name) -ForegroundColor Green
    } else {
        Write-Host ("[MISSING] {0}" -f $t.Name) -ForegroundColor Yellow
        $missing += $t
    }
}

$hasVC = Test-VCBuildTools
if ($hasVC) {
    Write-Host "[OK]      Visual Studio C++ Build Tools (needed by Tauri/Rust)" -ForegroundColor Green
} else {
    Write-Host "[MISSING] Visual Studio C++ Build Tools (needed by Tauri/Rust)" -ForegroundColor Yellow
}

if (Test-Command "claude") {
    Write-Host "[OK]      Claude Code" -ForegroundColor Green
} else {
    Write-Host "[MISSING] Claude Code (install it from the official docs: https://docs.claude.com/en/docs/claude-code)" -ForegroundColor Yellow
}

if (-not $Install) {
    Write-Host ""
    if ($missing.Count -eq 0 -and $hasVC) {
        Write-Host "Everything needed is installed." -ForegroundColor Green
    } else {
        Write-Host "Run again with -Install to install the missing tools." -ForegroundColor Cyan
    }
    exit 0
}

Write-Host ""
Write-Host "Installing missing tools..." -ForegroundColor Cyan
foreach ($t in $missing) {
    Write-Host ("--> {0}" -f $t.Name)
    try { Install-WithWinget $t.Id } catch { Write-Host "    failed: $_" -ForegroundColor Red }
}

if (-not $hasVC) {
    Write-Host "--> Visual Studio C++ Build Tools (large download, several GB)"
    try {
        Install-WithWinget "Microsoft.VisualStudio.2022.BuildTools" @(
            "--override",
            "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        )
    } catch { Write-Host "    failed: $_" -ForegroundColor Red }
}

if (Test-Command "ollama") {
    Write-Host "--> Pulling Ollama model $OllamaModel (about 3 GB)"
    try { & ollama pull $OllamaModel } catch { Write-Host "    failed: $_ (start the Ollama app and retry)" -ForegroundColor Red }
} else {
    Write-Host "Ollama was just installed: reopen the terminal, start Ollama, then run 'ollama pull $OllamaModel'." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Done. Close and reopen the terminal, then run this script again without -Install to verify." -ForegroundColor Cyan
