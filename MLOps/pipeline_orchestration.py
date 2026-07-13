#!/usr/bin/env python
"""
pipeline_orchestration.py — Roda o pipeline completo: dado bruto -> ABT -> predição.

Executa em sequência, cada etapa como um processo separado (mesmos scripts do
grupo, sem alterá-los):
    1. DataPipeline/data_sanitization.py   (Dados/raw/*.csv -> Dados/clean_data.csv)
    2. DataPipeline/abt_transform.py       (Dados/clean_data.csv -> Dados/abt.csv)
    3. Model/predict.py                    (Dados/abt.csv -> predicoes.csv)

Se uma etapa falhar (código de saída != 0), o pipeline para e reporta o erro —
não faz sentido gerar a ABT em cima de dados sujos, por exemplo.

Depois das etapas 1 e 2, o resultado também é copiado para uma pasta com o
nome da camada (arquitetura raw/silver/gold do MBA) — os arquivos originais
continuam nos caminhos de sempre, a cópia é só organização. Os nomes das
pastas vêm de pipeline_config.json.

Uso:
    python MLOps/pipeline_orchestration.py
    python MLOps/pipeline_orchestration.py --output predicoes.csv --skip-sanitization
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CFG = json.load(fh)


def organize_layer(layer: str, src_file: str, dest_name: str) -> None:
    """Copia src_file para Dados/<layer>/dest_name (cópia, não move o original)."""
    dest_dir = PROJECT_ROOT / CFG["layers"][layer]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_name
    shutil.copy2(PROJECT_ROOT / src_file, dest_path)
    logger.info("Camada '%s' organizada: %s", layer, dest_path)


def run_step(name: str, cmd: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("INICIANDO: %s", name)
    logger.info("Comando: %s", " ".join(cmd))
    logger.info("=" * 60)

    start = time.monotonic()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        logger.error("FALHOU: %s (código %s, %.1fs)", name, result.returncode, elapsed)
        sys.exit(result.returncode)

    logger.info("CONCLUÍDO: %s (%.1fs)", name, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo: bruto -> ABT -> predição.")
    parser.add_argument("--output", default="predicoes.csv", help="CSV de saída das predições")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de decisão")
    parser.add_argument("--skip-sanitization", action="store_true", help="Pula a limpeza (usa clean_data.csv existente)")
    parser.add_argument("--skip-abt", action="store_true", help="Pula a construção da ABT (usa abt.csv existente)")
    args = parser.parse_args()

    pipeline_start = time.monotonic()

    if not args.skip_sanitization:
        run_step("Sanitização dos dados", [sys.executable, "DataPipeline/data_sanitization.py"])
    else:
        logger.info("Sanitização pulada (--skip-sanitization)")
    organize_layer("silver", CFG["files"]["clean_data"], "clean_data.csv")

    if not args.skip_abt:
        run_step("Construção da ABT", [sys.executable, "DataPipeline/abt_transform.py"])
    else:
        logger.info("Construção da ABT pulada (--skip-abt)")
    organize_layer("gold", CFG["files"]["abt"], "abt.csv")

    run_step(
        "Predição em lote",
        [sys.executable, "Model/predict.py", "--input", "Dados/abt.csv", "--output", args.output, "--threshold", str(args.threshold)],
    )

    total_elapsed = time.monotonic() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETO em %.1fs — predições em %s", total_elapsed, args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
