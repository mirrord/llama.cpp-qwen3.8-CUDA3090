<#
.SYNOPSIS
    Launch the specialized text-only Qwen3.8-27B (IQ3_S) llama-server on RTX 3090.
.DESCRIPTION
    Serves the Unsloth Qwen3.8-27B-UD-IQ3_S GGUF with Q8_0 K/V cache, Flash Attention, full
    GPU offload. Two profiles via -Speculative:
      default        : 262144 context, temp 1.0 (full quality).
      -Speculative   : 196608 context, temp 0.7, MTP self-speculative decoding (faster gen).
    Binds to 127.0.0.1 only. This is a text-only build; --mmproj is refused.
#>
[CmdletBinding()]
param(
    [string]$ModelPath = "C:\Users\19082\.cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-IQ3_S.gguf",
    [string]$ServerExe = "",
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8080,
    [int]$CtxSize = 0,
    [int]$UBatch = 1024,
    [int]$Threads = 8,
    [int]$NGpuLayers = 99,
    [switch]$Speculative,
    [int]$SpecDraftNMax = 2,
    [double]$Temperature = 1.0
)

$ErrorActionPreference = "Stop"

# Unified runtime profile. -Speculative selects the fast profile: a smaller context that leaves
# room for the MTP draft context (~2 GiB at full context), plus a lower default temperature for
# higher draft acceptance. Without it, the full-quality profile (262144 context, temp 1.0) is used.
# Explicit -CtxSize / -Temperature override these profile defaults.
if ($CtxSize -le 0)     { $CtxSize     = if ($Speculative) { 196608 } else { 262144 } }
# if ($Temperature -lt 0) { $Temperature = if ($Speculative) { 0.7 }    else { 1.0 } }
$TempArg = $Temperature.ToString([System.Globalization.CultureInfo]::InvariantCulture)

# $PSScriptRoot is empty inside the param() default under [CmdletBinding()], so resolve here.
if (-not $ServerExe) {
    $ServerExe = Join-Path $PSScriptRoot "..\build-qwen38-prod\bin\llama-server.exe"
}

# fall back to the baseline build if the production binary is not present
if (-not (Test-Path -LiteralPath $ServerExe)) {
    $fallback = Join-Path $PSScriptRoot "..\build-qwen38-baseline\bin\llama-server.exe"
    if (Test-Path -LiteralPath $fallback) {
        $ServerExe = $fallback
    } else {
        throw "llama-server.exe not found. Build it first (preset qwen38-prod or qwen38-baseline). Looked for: $ServerExe"
    }
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model not found: $ModelPath"
}

# CUDA 13.4 runtime DLLs (cudart64_13, cublas64_13, cublasLt64_13) must be reachable.
# CUDA 13 relocated these from the toolkit bin dir to bin\x64.
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
if (Test-Path -LiteralPath $cudaBin) {
    $env:PATH = "$cudaBin;$env:PATH"
}

$serverArgs = @(
    "-m", $ModelPath,
    "--host", $ListenHost,
    "--port", $Port,
    "--ctx-size", $CtxSize,
    "--parallel", 1,
    "--flash-attn", "on",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--n-gpu-layers", $NGpuLayers,
    "--fit", "off",
    "--batch-size", 2048,
    "--ubatch-size", $UBatch,
    "--threads", $Threads,
    "--temp", $TempArg,
    "--no-context-shift",
    "--jinja",
    "--no-webui"
)

# Self-speculative decoding via the model's built-in MTP/NextN head (no separate draft model).
# Measured ~1.45x generation at temp 0.7-1.0 (~63% draft accept), ~1.63x at temp 0 (77% accept).
if ($Speculative) {
    $serverArgs += @(
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", $SpecDraftNMax,
        "--spec-draft-type-k", "q8_0",
        "--spec-draft-type-v", "q8_0"
    )
    if ($CtxSize -gt 196608) {
        Write-Warning "Speculative at ctx $CtxSize leaves little VRAM free (~191 MiB at 262144). Use -CtxSize 196608 or lower for a safe ~2.5 GiB headroom."
    }
}

Write-Host "Starting Qwen3.8-27B text-only server on http://$ListenHost`:$Port" -ForegroundColor Cyan
Write-Host "  model : $ModelPath"
Write-Host "  exe   : $ServerExe"
Write-Host "  ctx   : $CtxSize  kv: q8_0  fa: on  ngl: $NGpuLayers  ub: $UBatch  threads: $Threads"
Write-Host "  temp  : $TempArg"
Write-Host "  spec  : $(if ($Speculative) { "draft-mtp (n-max $SpecDraftNMax)" } else { 'off' })"

# llama-server logs to stderr; do not let PowerShell treat that as a terminating error
$ErrorActionPreference = "Continue"
& $ServerExe @serverArgs
exit $LASTEXITCODE
