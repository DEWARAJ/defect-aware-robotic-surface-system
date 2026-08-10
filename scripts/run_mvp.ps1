param(
    [string]$PythonExecutable = "python",
    [string]$RunDirectory = "runs/mvp"
)

& $PythonExecutable -m surface_perception.cli run-all --workspace $RunDirectory --config "configs/mvp.json"
if ($LASTEXITCODE -ne 0) {
    throw "MVP pipeline failed with exit code $LASTEXITCODE"
}

