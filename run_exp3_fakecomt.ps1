# Experimento 3 — Fakecom-t Defense (Cross-Prompt)
# Attack: completion_realcmb | Defense: none + defense_cross_prompt
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  EXP 3 - Fakecom-t Defense (Cross-Prompt)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python run_evaluation_instruction.py `
    --model_path "ollama:qwen2.5:7b" `
    --data_path "data/nllb_dataset_traduzido_pt.json" `
    --attack "completion_realcmb" `
    --defense "none" `
    --defense_cross_prompt `
    --log_path "logs/nllb_pt_base_qwen_results3.txt"

Write-Host "`n Log salvo em: logs/nllb_pt_base_qwen_results3.txt" -ForegroundColor Green
