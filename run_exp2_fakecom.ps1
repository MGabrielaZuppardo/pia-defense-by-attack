# Experimento 2 — Fakecom Defense
# Attack: completion_realcmb | Defense: injection-completionreal
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  EXP 2 — Fakecom Defense" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python run_evaluation_instruction.py `
    --model_path ollama:llama3.1:8b-instruct-q8_0 `
    --data_path data/crafted_instruction_data_davinci.json `
    --attack completion_realcmb `
    --defense injection-completionreal `
    --log_path logs/exp2_fakecom.txt

Write-Host "`n Log salvo em: logs/exp2_fakecom.txt" -ForegroundColor Green
