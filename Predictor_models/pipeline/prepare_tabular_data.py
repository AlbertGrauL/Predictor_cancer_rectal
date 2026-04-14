from __future__ import annotations

import argparse

from .config import load_config
from .tabular_utils import prepare_tabular_dataframe, split_dataframe
from .utils import load_paths, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera el manifiesto tabular y los splits reproducibles.")
    parser.add_argument("--config", default="Predictor_models/configs/tabular_baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    artifacts = prepare_tabular_dataframe(config)
    prepared = split_dataframe(artifacts.dataframe, config)

    manifest_columns = [
        "row_index",
        config["dataset"]["id_column"],
        "age",
        "sex",
        *config["dataset"]["binary_columns"],
        *config["dataset"]["ordinal_columns"],
        "digestive_family_history",
        "digestive_family_history_group",
        *[column for column in artifacts.feature_columns if column.startswith("digestive_family_history_")],
        config["dataset"]["target_column"],
        "split",
    ]
    manifest_rows = prepared[manifest_columns].to_dict(orient="records")
    manifest_path = paths.manifests_dir / "tabular_manifest.csv"
    write_csv(manifest_path, manifest_rows, manifest_columns)

    split_summary = {}
    for split_name, split_df in prepared.groupby("split"):
        split_summary[split_name] = {
            "rows": int(len(split_df)),
            "sin_riesgo_clinico": int((split_df[config["dataset"]["target_column"]] == 0).sum()),
            "riesgo_clinico": int((split_df[config["dataset"]["target_column"]] == 1).sum()),
        }

    split_summary_path = paths.reports_dir / "tabular_split_summary.json"
    write_json(split_summary_path, split_summary)
    print(f"Manifiesto tabular guardado en: {manifest_path}")
    print(f"Resumen de splits guardado en: {split_summary_path}")


if __name__ == "__main__":
    main()
