param(
    [string]$PythonExecutable = "python",
    [string]$Model = "runs/mvtec_v04/model_128/global_model.onnx",
    [string]$Manifest = "runs/mvtec_v04/manifest.json",
    [string]$TrainingReport = "runs/mvtec_v04/model_128/training_report.json",
    [string]$Output = "runs/mvtec_v06_int8",
    [int]$CalibrationSamples = 64,
    [int]$Iterations = 200,
    [int]$Trials = 5
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & $PythonExecutable -m surface_perception.quantize_onnx `
        --model $Model `
        --manifest $Manifest `
        --training-report $TrainingReport `
        --output $Output `
        --image-size 128 `
        --calibration-samples $CalibrationSamples `
        --iterations $Iterations `
        --warmup 20 `
        --trials $Trials
    if ($LASTEXITCODE -ne 0) {
        throw "INT8 experiment failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
