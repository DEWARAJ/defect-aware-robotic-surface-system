param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$PythonExecutable = "python",
    [string]$OutputDirectory = "runs/production-contract",
    [string]$Config = "configs/production_experiment_v07.json",
    [switch]$AllowDirty
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$inventory = Join-Path $OutputDirectory "dataset_inventory.json"
$contract = Join-Path $OutputDirectory "experiment_contract.json"

Push-Location $projectRoot
try {
    & $PythonExecutable -m surface_perception.cli inventory `
        --manifest $Manifest --output $inventory
    if ($LASTEXITCODE -ne 0) { throw "Dataset inventory creation failed" }

    & $PythonExecutable -m surface_perception.cli verify-inventory `
        --manifest $Manifest --inventory $inventory
    if ($LASTEXITCODE -ne 0) { throw "Dataset inventory verification failed" }

    $arguments = @(
        "-m", "surface_perception.cli", "experiment-contract",
        "--config", $Config,
        "--inventory", $inventory,
        "--output", $contract,
        "--repository-root", "."
    )
    if ($AllowDirty) { $arguments += "--allow-dirty" }
    & $PythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment contract creation failed" }
}
finally {
    Pop-Location
}
