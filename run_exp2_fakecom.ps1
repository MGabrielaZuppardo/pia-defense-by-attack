# Experimento 2 — Fakecom Defense
# Attack: completion_realcmb | Defense: injection-completionreal
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  EXP 2 - Fakecom Defense" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python run_evaluation_instruction.py `
    --model_path "ollama:qwen2.5:7b" `
    --data_path "data/nllb_dataset_traduzido_pt.json" `
    --attack "completion_realcmb" `
    --defense "injection-completionreal" `
    --log_path "logs/nllb_pt_base_qwen_results2.txt"

Write-Host ""
Write-Host "Log salvo em: logs/nllb_pt_base_qwen_results2.txt" -ForegroundColor Green