# Experimento 1 — Sem Defesa (Baseline)
# Attack: completion_realcmb | Defense: none
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  EXP 1 — Baseline (Sem Defesa)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python run_evaluation_instruction.py `
    --model_path ollama:llama3.1:8b-instruct-q8_0 `
    --data_path data/crafted_instruction_data_davinci.json `
    --attack completion_realcmb `
    --defense none `
    --log_path logs/exp1_baseline.txt

Write-Host "`n Log salvo em: logs/exp1_baseline.txt" -ForegroundColor Green
