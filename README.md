# Avaliação dos Impactos de Prompt Injection em Português

Desenvolvido por **Lucas Albuquerque Lisboa** e **Maria Gabriela Alves Zuppardo**.

Repositório do projeto da disciplia de _Tópicos Avançados em Inteligência Computacional 1 (Deep Learning)_ ministrada pelo professor Cleber Zanchettin.

Este projeto adapta o código do artigo [_Defense Against Prompt Injection Attack by Leveraging Attack Techniques_](https://arxiv.org/abs/2411.00459), disponível em: https://github.com/LukeChen-go/pia-defense-by-attack .


## Ambiente

Para executação em ambiente local, recomenda-se instalação do Ollama e dos modelos que deseja avaliar. Para instalação das depedências, segue comando extraído do repositório original: 

```
conda creat -n defenseinj python=3.12
conda activate defenseinj
pip install -r requirements.txt

```

## Avaliação

Para executar um experimento de avaliação, deve-se realizar o comando no seguinte formato:

```angular2html
python run_evaluation_instruction.py \
--defense Nome_da_Defesa \
--attack Nome_do_Ataque \
--log_path logs/Nome_do_Log.txt \ 
--data_path data/Nome_da_Base.json \ 
--model_path Nome_do_Modelo

```
No ínicio da do comando, há `python run_evaluation_instruction.py`, seguido de `--defense Nome_da_Defesa` (substituir Nome_da_Defesa pela defesa correspondente), seguido de `--attack Nome_do_Ataque` (substituir Nome_do_Ataque pela ataque correspondente), seguido de `--log_path logs/Nome_do_Log.txt` (substituir Nome_do_Log pelo nome que deseja identificar o nome do arquivo mantendo formato .txt), seguido de `--data_path data/Nome_da_Base.json` (substituir Nome_da_Base pela base correspondente mantendo formato .json) e seguido de `--model_path Nome_do_Modelo` (substituir Nome_do_Modelo pelo modelo correspondente). Desse modo, segue exemplo prático extraído do repositório original:

```angular2html
python run_evaluation_instruction.py \
--defense injection-completionreal \
--attack completion_realcmb \
--log_path logs/defense_llama3_inst.txt \ 
--data_path data/crafted_instruction_data_qa.json \ 
--model_path meta-llama/Meta-Llama-3-8B-Instruct

```