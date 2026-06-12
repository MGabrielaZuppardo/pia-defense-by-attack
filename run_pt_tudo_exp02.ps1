# Experimentos exp02 (fakecom) — instrução PT, base PT, injeção PT
# Roda sequencialmente para os 3 modelos

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$experiments = @(
    @{
        model    = "ollama:qwen2.5:7b"
        log      = "logs/qwen25_7b/pt_tudo/exp02_qa_pt_tudo_fakecom_qwen25_7b.txt"
        label    = "qwen2.5:7b"
    },
    @{
        model    = "ollama:llama3.1:8b-instruct-q8_0"
        log      = "logs/llama31_8b/pt_tudo/exp02_qa_pt_tudo_fakecom_llama31_8b.txt"
        label    = "llama3.1:8b-instruct-q8_0"
    },
    @{
        model    = "ollama:tinyllama:latest"
        log      = "logs/tinyllama_1b/pt_tudo/exp02_qa_pt_tudo_fakecom_tinyllama.txt"
        label    = "tinyllama:latest"
    }
)

$data = "data/crafted_instruction_data_qa_pt.json"

foreach ($exp in $experiments) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Modelo: $($exp.label)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    python run_evaluation_instruction_pt_inj.py `
        --model_path $exp.model `
        --data_path  $data `
        --attack     completion_realcmb `
        --defense    injection-completionreal `
        --log_path   $exp.log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRO no modelo $($exp.label) — abortando." -ForegroundColor Red
        exit 1
    }

    Write-Host "Concluido: $($exp.label)" -ForegroundColor Green
}

Write-Host "`nTodos os exp02 pt_tudo concluidos." -ForegroundColor Green
