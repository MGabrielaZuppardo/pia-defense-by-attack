# exp01 (sem defesa) — instrução EN, base EN, injeção PT — apenas llama e tinyllama

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$data = "data/crafted_instruction_data_qa.json"

$experiments = @(
    @{
        model = "ollama:llama3.1:8b-instruct-q8_0"
        log   = "logs/llama31_8b/pt_so_injecao/exp01_qa_pt_so_inj_sem_defesa_llama31_8b.txt"
        label = "llama3.1:8b-instruct-q8_0"
    },
    @{
        model = "ollama:tinyllama:latest"
        log   = "logs/tinyllama_1b/pt_so_injecao/exp01_qa_pt_so_inj_sem_defesa_tinyllama.txt"
        label = "tinyllama:latest"
    }
)

foreach ($exp in $experiments) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Modelo: $($exp.label)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    python run_evaluation_instruction_pt_inj.py `
        --model_path $exp.model `
        --data_path  $data `
        --attack     completion_realcmb `
        --defense    none `
        --log_path   $exp.log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRO no modelo $($exp.label) — abortando." -ForegroundColor Red
        exit 1
    }

    Write-Host "Concluido: $($exp.label)" -ForegroundColor Green
}

Write-Host "`nLlama e Tinyllama exp01 concluidos." -ForegroundColor Green
