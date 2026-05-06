"""Build a single index.html under `results/<run>/` linking every PNG produced by stage 3."""

import os
import sys
import html

from veloline import io_state


def main():
    run_dir = io_state.latest_run_dir() if len(sys.argv) < 2 else sys.argv[1]
    plots_root = os.path.join(run_dir, "plots")
    if not os.path.isdir(plots_root):
        print(f"[make_index_html] no plots/ under {run_dir}")
        sys.exit(1)

    sections = sorted(d for d in os.listdir(plots_root) if os.path.isdir(os.path.join(plots_root, d)))
    out_lines = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
                 f"<title>{html.escape(os.path.basename(run_dir))}</title>",
                 "<style>body{font-family:sans-serif;max-width:1200px;margin:auto}"
                 "img{max-width:100%;border:1px solid #ddd;margin:8px 0}"
                 "h1{color:#333}h2{margin-top:2em;color:#555}</style></head><body>",
                 f"<h1>{html.escape(os.path.basename(run_dir))}</h1>"]
    for sec in sections:
        out_lines.append(f"<h2>{html.escape(sec)}</h2>")
        sec_dir = os.path.join(plots_root, sec)
        for png in sorted(p for p in os.listdir(sec_dir) if p.lower().endswith(".png")):
            rel = f"plots/{sec}/{png}"
            out_lines.append(f"<div><div><code>{html.escape(rel)}</code></div>"
                             f"<img src='{html.escape(rel)}' alt='{html.escape(png)}'></div>")

    metrics_root = os.path.join(run_dir, "metrics")
    if os.path.isdir(metrics_root):
        out_lines.append("<h2>metrics</h2><ul>")
        for f in sorted(os.listdir(metrics_root)):
            out_lines.append(f"<li><a href='metrics/{html.escape(f)}'>{html.escape(f)}</a></li>")
        out_lines.append("</ul>")
    out_lines.append("</body></html>")

    out_path = os.path.join(run_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"[make_index_html] wrote {out_path}")


if __name__ == "__main__":
    main()
