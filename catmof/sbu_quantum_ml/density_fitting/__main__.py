"""CLI entry: ``python -m catmof.sbu_quantum_ml.density_fitting …``."""

from __future__ import annotations

import argparse
from pathlib import Path

from catmof.sbu_quantum_ml.density_fitting.assemble_features import (
    assemble_class_tiled_features,
    merge_per_sbu_pickles,
)
from catmof.sbu_quantum_ml.density_fitting.xtb_parse import parse_xtb_root_to_dataframe


def main() -> None:
    p = argparse.ArgumentParser(description="SBU density-fitting / xTB helper CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_asm = sub.add_parser("assemble-features", help="Merge coeff bundle + xTB CSV → density_fitting_features.pkl")
    p_asm.add_argument("--coeff-pkl", type=Path, required=True)
    p_asm.add_argument("--xtb-csv", type=Path, required=True)
    p_asm.add_argument("--out", type=Path, default=Path("density_fitting_features.pkl"))
    p_asm.add_argument("--xtb-sbu-col", type=str, default="sbu")

    p_merge = sub.add_parser("merge-df-pickles", help="Concatenate per-SBU *_df_e3nn.pkl into one bundle")
    p_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_merge.add_argument("--out", type=Path, required=True)

    p_xtb = sub.add_parser("xtb-parse-root", help="Parse xTB logs under a root directory → CSV")
    p_xtb.add_argument("--root", type=Path, required=True)
    p_xtb.add_argument("--out-csv", type=Path, default=Path("xtb_output_data.csv"))

    args = p.parse_args()

    if args.cmd == "assemble-features":
        assemble_class_tiled_features(
            args.coeff_pkl,
            args.xtb_csv,
            args.out,
            sbu_column_xtb=args.xtb_sbu_col,
        )
        print(f"Wrote {args.out}")
    elif args.cmd == "merge-df-pickles":
        merge_per_sbu_pickles(args.inputs, args.out)
        print(f"Wrote {args.out}")
    elif args.cmd == "xtb-parse-root":
        df = parse_xtb_root_to_dataframe(args.root)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
