param(
    [string]$PythonExecutable = "python",
    [string]$Config = "configs/active_learning_v11.json",
    [string]$Manifest = "",
    [string]$Checkpoint = "",
    [string]$Output = "",
    [string]$PortableEvidence = ""
)

$arguments = @("-m", "surface_perception.active_learning", "--config", $Config)
if ($Manifest) {
    $arguments += @("--manifest", $Manifest)
}
if ($Checkpoint) {
    $arguments += @("--checkpoint", $Checkpoint)
}
if ($Output) {
    $arguments += @("--output", $Output)
}
if ($PortableEvidence) {
    $arguments += @("--portable-evidence", $PortableEvidence)
}

& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) {
    throw "active-learning study failed with exit code $LASTEXITCODE"
}
