param(
    [string]$PythonExecutable = "python",
    [string]$OutputDirectory = "runs/coverage_demo"
)

& $PythonExecutable -m surface_perception.cli plan-demo --output $OutputDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Coverage demonstration failed with exit code $LASTEXITCODE"
}
