"""
Traduz crafted_instruction_data_qa.json (EN) para PT usando Google Translate.
Traduz os campos: instruction, input, output.
Salva em data/crafted_instruction_data_qa_googletrans_pt.json com checkpoint.

Uso:
    python translate_dataset.py
    python translate_dataset.py --start 500        # retoma do item 500
    python translate_dataset.py --limit 100        # traduz só 100 itens (teste)
    python translate_dataset.py --fields instruction output  # traduz só esses campos
"""

import argparse
import json
import os
import time
from deep_translator import GoogleTranslator

SOURCE_FILE = "data/crafted_instruction_data_qa.json"
OUTPUT_FILE = "data/crafted_instruction_data_qa_googletrans_pt.json"
CHECKPOINT_FILE = "data/crafted_instruction_data_qa_googletrans_pt.checkpoint.json"
FIELDS = ["instruction", "input", "output"]
BATCH_SAVE = 50  # salva checkpoint a cada N itens


def translate(text: str, translator: GoogleTranslator) -> str:
    if not text or not text.strip():
        return text
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"  [WARN] Falha na tradução: {e}. Mantendo original.")
        return text


def load_checkpoint() -> list:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(data: list):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--start", type=int, default=None,
                        help="Índice para retomar (padrão: continua do checkpoint)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Número máximo de itens a traduzir")
    parser.add_argument("--fields", nargs="+", default=FIELDS,
                        help="Campos a traduzir (padrão: instruction input output)")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="pt")
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as f:
        data = json.load(f)

    translated = load_checkpoint()
    start = args.start if args.start is not None else len(translated)

    if start > 0:
        print(f"Retomando do item {start} ({len(translated)} já traduzidos).")

    end = min(start + args.limit, len(data)) if args.limit else len(data)
    total = end - start

    print(f"Traduzindo {total} itens ({start} -> {end}) | campos: {args.fields}")
    print(f"Idioma: {args.src_lang} -> {args.tgt_lang}\n")

    translator = GoogleTranslator(source=args.src_lang, target=args.tgt_lang)

    for i, item in enumerate(data[start:end], start=start):
        new_item = dict(item)
        for field in args.fields:
            if field in item and item[field]:
                new_item[field] = translate(item[field], translator)

        translated.append(new_item)

        done = i - start + 1
        if done % 10 == 0:
            print(f"  [{done}/{total}] item {i} — instruction: {new_item.get('instruction','')[:60]}...")

        if done % BATCH_SAVE == 0:
            save_checkpoint(translated)
            print(f"  >> Checkpoint salvo ({len(translated)} itens)")

        # pequeno delay para não sobrecarregar a API
        time.sleep(0.05)

    # salva resultado final
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    # remove checkpoint se concluiu tudo
    if len(translated) >= len(data) and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removido (tradução completa).")

    print(f"\nPronto. {len(translated)} itens salvos em '{args.output}'.")


if __name__ == "__main__":
    main()
