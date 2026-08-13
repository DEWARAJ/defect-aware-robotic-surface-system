param(
    [string]$PythonExecutable = "python",
    [string]$Config = "configs/model_compression_v10.json",
    [string]$Manifest = "",
    [string]$TeacherCheckpoint = "",
    [string]$Output = ""
)

$arguments = @("-m", "surface_perception.model_compression", "--config", $Config)
if ($Manifest) {
    $arguments += @("--manifest", $Manifest)
}
if ($TeacherCheckpoint) {
    $arguments += @("--teacher-checkpoint", $TeacherCheckpoint)
}
if ($Output) {
    $arguments += @("--output", $Output)
}

& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) {
    throw "model compression study failed with exit code $LASTEXITCODE"
}
