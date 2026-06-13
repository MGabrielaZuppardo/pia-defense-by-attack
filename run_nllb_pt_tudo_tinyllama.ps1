# Experimentos exp01, exp02, exp03 — instrução PT, base PT, injeção PT — dataset NLLB
# Apenas tinyllama

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$data = "data/nllb_dataset_traduzido_pt.json"
$model = "ollama:tinyllama:latest"

$experiments = @(
    @{
        label   = "exp01 (sem defesa)"
        defense = "none"
        cross   = $false
        log     = "logs/tinyllama_1b/nllb_pt_tudo/exp01_qa_nllb_pt_tudo_sem_defesa_tinyllama.txt"
    },
    @{
        label   = "exp02 (fakecom)"
        defense = "injection-completionreal"
        cross   = $false
        log     = "logs/tinyllama_1b/nllb_pt_tudo/exp02_qa_nllb_pt_tudo_fakecom_tinyllama.txt"
    },
    @{
        label   = "exp03 (fakecomt)"
        defense = "none"
        cross   = $true
        log     = "logs/tinyllama_1b/nllb_pt_tudo/exp03_qa_nllb_pt_tudo_fakecomt_tinyllama.txt"
    }
)

foreach ($exp in $experiments) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $($exp.label)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    if ($exp.cross) {
        python run_evaluation_instruction_pt_inj.py `
            --model_path $model `
            --data_path  $data `
            --attack     completion_realcmb `
            --defense    $exp.defense `
            --defense_cross_prompt `
            --log_path   $exp.log
    } else {
        python run_evaluation_instruction_pt_inj.py `
            --model_path $model `
            --data_path  $data `
            --attack     completion_realcmb `
            --defense    $exp.defense `
            --log_path   $exp.log
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRO em $($exp.label) — abortando." -ForegroundColor Red
        exit 1
    }

    Write-Host "Concluido: $($exp.label)" -ForegroundColor Green
}

Write-Host "`nTodos os 3 experimentos tinyllama nllb_pt_tudo concluidos." -ForegroundColor Green
