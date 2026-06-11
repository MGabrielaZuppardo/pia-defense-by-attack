"""
Traduz as sentenças de injeção (IGNORE_ATTACK_SENTENCES) de instruction_attack_defense_tools.py
para português e atualiza instruction_attack_defense_tools_pt_inj.py.

Uso:
    python translate_injections.py
    python translate_injections.py --source en --target pt  # padrão
    python translate_injections.py --dry-run               # só imprime, não salva
"""

import argparse
import ast
import re
import sys
from deep_translator import GoogleTranslator


def extract_attack_sentences(source_path: str) -> dict[str, list[str]]:
    """Lê IGNORE_ATTACK_SENTENCES do arquivo fonte via AST (não executa o módulo)."""
    with open(source_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IGNORE_ATTACK_SENTENCES":
                    return ast.literal_eval(node.value)

    raise ValueError("IGNORE_ATTACK_SENTENCES não encontrado em " + source_path)


def translate_sentences(sentences: dict[str, list[str]], src: str, tgt: str) -> dict[str, list[str]]:
    """Traduz cada sentença preservando o placeholder {injected_prompt}."""
    translator = GoogleTranslator(source=src, target=tgt)
    result = {}

    for split, items in sentences.items():
        translated = []
        for sentence in items:
            # Remove o placeholder antes de traduzir para ele não ser alterado
            placeholder = "{injected_prompt}"
            safe = sentence.replace(placeholder, "XPLACEHOLDERX")
            t = translator.translate(safe)
            # Restaura o placeholder
            t = t.replace("XPLACEHOLDERX", placeholder)
            # Google às vezes capitaliza, corrige para manter padrão original
            if sentence[0].islower():
                t = t[0].lower() + t[1:]
            translated.append(t)
            print(f"  [{split}] {sentence[:60]}...")
            print(f"       > {t[:60]}...")
        result[split] = translated

    return result


def patch_file(target_path: str, translated: dict[str, list[str]]) -> str:
    """Substitui o bloco IGNORE_ATTACK_SENTENCES no arquivo alvo."""
    with open(target_path, encoding="utf-8") as f:
        content = f.read()

    # Localiza o início da atribuição
    start_match = re.search(r"IGNORE_ATTACK_SENTENCES\s*=\s*\{", content)
    if not start_match:
        raise ValueError("IGNORE_ATTACK_SENTENCES não encontrado em " + target_path)

    # Avança contando chaves para encontrar o fechamento correto
    brace_start = content.index("{", start_match.start())
    depth = 0
    in_single = False
    in_double = False
    i = brace_start
    while i < len(content):
        ch = content[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        i += 1

    # Monta o novo bloco usando aspas duplas para evitar conflitos
    new_block = "IGNORE_ATTACK_SENTENCES = {\n"
    for split, items in translated.items():
        new_block += f'    "{split}": [\n'
        for s in items:
            escaped = s.replace("\\", "\\\\").replace('"', '\\"')
            new_block += f'        "{escaped}",\n'
        new_block += "    ],\n"
    new_block += "}"

    return content[:start_match.start()] + new_block + content[i + 1:]


def main():
    parser = argparse.ArgumentParser(description="Traduz sentenças de injeção para o idioma alvo.")
    parser.add_argument("--source-file", default="instruction_attack_defense_tools.py",
                        help="Arquivo fonte com as sentenças em inglês")
    parser.add_argument("--target-file", default="instruction_attack_defense_tools_pt_inj.py",
                        help="Arquivo alvo a ser atualizado")
    parser.add_argument("--source", default="en", help="Idioma de origem (padrão: en)")
    parser.add_argument("--target", default="pt", help="Idioma de destino (padrão: pt)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime as traduções sem salvar o arquivo")
    args = parser.parse_args()

    print(f"Extraindo IGNORE_ATTACK_SENTENCES de '{args.source_file}'...")
    sentences = extract_attack_sentences(args.source_file)
    total = sum(len(v) for v in sentences.values())
    print(f"Encontradas {total} sentenças em {list(sentences.keys())}.\n")

    print(f"Traduzindo {args.source} -> {args.target}...\n")
    translated = translate_sentences(sentences, args.source, args.target)

    if args.dry_run:
        print("\n--- Resultado (dry-run, nada foi salvo) ---")
        import json
        print(json.dumps(translated, ensure_ascii=False, indent=2))
        sys.exit(0)

    print(f"\nAtualizando '{args.target_file}'...")
    patched = patch_file(args.target_file, translated)
    with open(args.target_file, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"Arquivo '{args.target_file}' atualizado com sucesso.")


if __name__ == "__main__":
    main()
