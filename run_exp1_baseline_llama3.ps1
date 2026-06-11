Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "EXP 1 - Baseline (Sem Defesa)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python run_evaluation_instruction.py `
    --model_path "ollama:llama3.1:8b-instruct-q8_0" `
    --data_path "data\nllb_dataset_traduzido_pt.json" `
    --attack "completion_realcmb" `
    --defense "none" `
    --log_path "logs/nllb_pt_base_llama3_results1.txt"

Write-Host ""
Write-Host "Log salvo em: logs/nllb_pt_base_llama3_results1.txt" -ForegroundColor Green