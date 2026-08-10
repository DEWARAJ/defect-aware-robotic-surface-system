param(
    [Parameter(Mandatory = $true)]
    [string]$MvtecRoot,
    [string[]]$Categories = @("metal_nut"),
    [string]$PythonExecutable = "python",
    [string]$RunDirectory = "runs/real_data"
)

$manifest = Join-Path $RunDirectory "manifest.json"
$modelDirectory = Join-Path $RunDirectory "model"
$evaluationDirectory = Join-Path $RunDirectory "evaluation"
$onnxModel = Join-Path $modelDirectory "model.onnx"

& $PythonExecutable -m surface_perception.cli build-manifest `
    --mvtec-root $MvtecRoot --categories $Categories --output $manifest
if ($LASTEXITCODE -ne 0) { throw "Manifest creation failed" }

& $PythonExecutable -m surface_perception.train_real `
    --manifest $manifest --output $modelDirectory
if ($LASTEXITCODE -ne 0) { throw "Training failed" }

& $PythonExecutable -m surface_perception.evaluate_real `
    --manifest $manifest --checkpoint (Join-Path $modelDirectory "best.pt") `
    --output $evaluationDirectory
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }

& $PythonExecutable -m surface_perception.export_onnx `
    --checkpoint (Join-Path $modelDirectory "best.pt") --output $onnxModel
if ($LASTEXITCODE -ne 0) { throw "ONNX export failed" }

& $PythonExecutable -m surface_perception.onnx_parity `
    --checkpoint (Join-Path $modelDirectory "best.pt") --onnx $onnxModel `
    --manifest $manifest --output (Join-Path $modelDirectory "onnx_parity.json")
if ($LASTEXITCODE -ne 0) { throw "ONNX parity verification failed" }

& $PythonExecutable -m surface_perception.benchmark_onnx `
    --model $onnxModel --output (Join-Path $modelDirectory "onnx_benchmark.json")
if ($LASTEXITCODE -ne 0) { throw "ONNX benchmark failed" }
