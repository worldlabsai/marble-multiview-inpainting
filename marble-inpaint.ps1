$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "marble-inpaint needs uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 127
}

uv run --project $PSScriptRoot --frozen --no-dev marble-inpaint @args
exit $LASTEXITCODE
