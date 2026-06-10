# Experimento 3 — Fakecom-t Defense (Cross-Prompt)
# Attack: completion_realcmb | Defense: none + defense_cross_prompt
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  EXP 3 — Fakecom-t Defense (Cross-Prompt)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python run_evaluation_instruction.py `
    --model_path ollama:llama3.1:8b-instruct-q8_0 `
    --data_path data/crafted_instruction_data_davinci.json `
    --attack completion_realcmb `
    --defense none `
    --defense_cross_prompt `
    --log_path logs/exp3_fakecomt.txt

Write-Host "`n Log salvo em: logs/exp3_fakecomt.txt" -ForegroundColor Green
