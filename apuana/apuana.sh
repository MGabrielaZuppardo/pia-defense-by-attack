#!/bin/bash
#SBATCH --job-name=Lucas_CPFL
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH --cpus-per-task=16
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lal3@cin.ufpe.br
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

# carregar versão python
module load Python/3.12
# ativar ambiente
source $HOME/pia-defense-by-attack/bin/activate
# executar .py
python $HOME/pia-defense-by-attack/run_exp1_baseline.ps1