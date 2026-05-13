#!/usr/bin/env python
import argparse
import subprocess
import sys


def build_run10x_command(args):
    if not args.sample_dir or not args.gtf:
        raise ValueError("--sample-dir and --gtf are required for mode=run10x")

    cmd = ["velocyto", "run10x"]
    if args.mask:
        cmd += ["-m", args.mask]
    if args.outdir:
        cmd += ["-o", args.outdir]
    cmd += [args.sample_dir, args.gtf]
    return cmd


def build_run_command(args):
    missing = [name for name in ["barcodes", "bam", "gtf", "outdir"] if getattr(args, name) is None]
    if missing:
        raise ValueError("Missing required args for mode=run: %s" % ", ".join(missing))

    cmd = ["velocyto", "run", "-b", args.barcodes, "-o", args.outdir]
    if args.mask:
        cmd += ["-m", args.mask]
    cmd += [args.bam, args.gtf]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Generate a loom file using velocyto.")
    parser.add_argument("--mode", choices=["run10x", "run"], default="run10x")
    parser.add_argument("--execute", action="store_true", help="Run the command instead of printing it")

    parser.add_argument("--sample-dir", help="10x output folder for run10x")
    parser.add_argument("--gtf", help="Reference GTF file")
    parser.add_argument("--mask", help="Repeat-mask GTF file")
    parser.add_argument("--outdir", help="Output directory for loom")

    parser.add_argument("--barcodes", help="Barcodes TSV (for mode=run)")
    parser.add_argument("--bam", help="Aligned BAM (for mode=run)")

    args = parser.parse_args()

    if args.mode == "run10x":
        cmd = build_run10x_command(args)
    else:
        cmd = build_run_command(args)

    print("[cmd] " + " ".join(cmd))

    if args.execute:
        subprocess.run(cmd, check=True)
        print("[ok] velocyto finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[error] %s" % exc)
        sys.exit(1)
