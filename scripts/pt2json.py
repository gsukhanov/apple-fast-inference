import argparse
from pathlib import Path

import torch
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Конвертация pt в json формат")
    parser.add_argument("--file_path", type=Path, required=True)
    parser.add_argument(
        "--map_location",
        type=str,
        default="cpu",
        help="Устройства для torch.load (например: 'cpu', 'cuda:0'). По умолчанию: cpu",
    )
    args = parser.parse_args(argv)

    if not args.file_path.exists():
        parser.error(f"Файл не найден: {args.file_path}")

    model = torch.load(args.file_path, map_location=torch.device(args.map_location))
    if not hasattr(model, "items"):
        parser.error("Ожидался state_dict в .pt файле")

    data = {}
    for k, v in model.items():
        data[k] = v.tolist() if hasattr(v, "tolist") else v
    
    json_path = args.file_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
