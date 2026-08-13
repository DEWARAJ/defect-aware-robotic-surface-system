param(
    [Parameter(Mandatory = $true)]
    [string]$IsaacPython,
    [int]$Frames = 50,
    [string]$Output = "runs/isaac_sim_surface",
    [switch]$Headless
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$arguments = @(
    "scripts/isaac_sim/generate_surface_dataset.py",
    "--config", "configs/isaac_sim_surface.json",
    "--output", $Output,
    "--frames", $Frames
)
if ($Headless) {
    $arguments += "--headless"
}

Push-Location $projectRoot
try {
    & $IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Sim generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
