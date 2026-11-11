"""
build_pdf_report.py
Converts METHODOLOGY_REPORT.md into a self-contained, publication-ready PDF
using Microsoft Edge headless and inlined base64 images.
"""

import os
import base64
import markdown
import subprocess

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].replace(".", "")
            return f"data:image/{ext};base64,{encoded}"
    return ""

def main():
    print("Reading METHODOLOGY_REPORT.md...")
    with open("METHODOLOGY_REPORT.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'codehilite', 'nl2br']
    )

    # Base64 images
    img_b64_cal = get_base64_image("calibration_wp.png")
    img_b64_dec = get_base64_image("pressure_by_decile.png")
    img_b64_bat = get_base64_image("top_batters_pressure_matrix.png")
    img_b64_bwl = get_base64_image("top_bowlers_pressure_matrix.png")

    # Image section to insert into HTML
    figures_html = f"""
    <div class="page-break"></div>
    <h2>Visual Diagnostic Gallery & Performance Matrices</h2>
    
    <div class="figure-card">
      <h3>Figure 1: Top 15 Batters Under High Pressure (RAE Matrix)</h3>
      <img src="{img_b64_bat}" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" />
    </div>

    <div class="figure-card" style="margin-top: 25px;">
      <h3>Figure 2: Top 15 Bowlers Under High Pressure (RAE Matrix)</h3>
      <img src="{img_b64_bwl}" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" />
    </div>

    <div class="page-break"></div>

    <div class="figure-card">
      <h3>Figure 3: Win Probability Model Calibration (Innings 1 & 2)</h3>
      <img src="{img_b64_cal}" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" />
    </div>

    <div class="figure-card" style="margin-top: 25px;">
      <h3>Figure 4: Empirical Outcome Distributions by Pressure Decile</h3>
      <img src="{img_b64_dec}" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" />
    </div>
    """

    # Add figures right before Section 6 or at the end
    if "<h2>6. Generated Visualizations" in html_content:
        html_content = html_content.replace("<h2>6. Generated Visualizations", figures_html + "\n<h2>6. Generated Visualizations")
    else:
        html_content += figures_html

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SupeRR Selector Season 2: Methodology Report</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
      @bottom-right {{
        content: counter(page);
      }}
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 10.5pt;
      line-height: 1.55;
      color: #1a1a1a;
      background: #ffffff;
      margin: 0;
      padding: 0;
    }}
    h1 {{
      font-size: 20pt;
      color: #0D47A1;
      margin-top: 0;
      margin-bottom: 6px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }}
    h2 {{
      font-size: 13.5pt;
      color: #1565C0;
      margin-top: 22px;
      margin-bottom: 8px;
      border-bottom: 1.5px solid #E3F2FD;
      padding-bottom: 4px;
      font-weight: 700;
    }}
    h3 {{
      font-size: 11.5pt;
      color: #2E7D32;
      margin-top: 16px;
      margin-bottom: 6px;
      font-weight: 600;
    }}
    p {{
      margin: 6px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      font-size: 9pt;
      page-break-inside: avoid;
    }}
    th {{
      background-color: #F0F4F8;
      color: #0D47A1;
      font-weight: 700;
      text-align: left;
      padding: 6px 8px;
      border: 1px solid #CFD8DC;
    }}
    td {{
      padding: 5px 8px;
      border: 1px solid #CFD8DC;
      vertical-align: middle;
    }}
    tr:nth-child(even) {{
      background-color: #F8FAFC;
    }}
    code {{
      background-color: #F1F5F9;
      color: #D32F2F;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
      font-size: 8.8pt;
      padding: 1px 4px;
      border-radius: 3px;
    }}
    pre {{
      background-color: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-left: 3px solid #0D47A1;
      padding: 10px;
      font-family: Consolas, monospace;
      font-size: 8.2pt;
      line-height: 1.4;
      overflow-x: auto;
      page-break-inside: avoid;
      border-radius: 4px;
    }}
    .page-break {{
      page-break-before: always;
    }}
    .figure-card {{
      background: #FFFFFF;
      border: 1px solid #E2E8F0;
      border-radius: 8px;
      padding: 12px;
      page-break-inside: avoid;
    }}
    ul, ol {{
      margin: 6px 0 10px 20px;
      padding: 0;
    }}
    li {{
      margin-bottom: 4px;
    }}
    hr {{
      border: 0;
      height: 1px;
      background: #E2E8F0;
      margin: 18px 0;
    }}
  </style>
</head>
<body>
  {html_content}
</body>
</html>
"""

    report_html_path = "METHODOLOGY_REPORT.html"
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated standalone HTML report: {report_html_path}")

    # Compile to PDF using Edge headless
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf_out = "METHODOLOGY_REPORT.pdf"
    abs_html = os.path.abspath(report_html_path)
    abs_pdf = os.path.abspath(pdf_out)

    if os.path.exists(edge_path):
        print("Compiling PDF via Microsoft Edge headless...")
        cmd = [
            edge_path,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={abs_pdf}",
            abs_html
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(pdf_out):
            pdf_size_kb = os.path.getsize(pdf_out) / 1024
            print(f"[✓] Successfully compiled: {pdf_out} ({pdf_size_kb:.1f} KB)")
        else:
            print("Edge execution completed but PDF not found. Output:", res.stderr)
    else:
        print("Edge executable not found at specified path.")

if __name__ == "__main__":
    main()
