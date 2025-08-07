from pathlib import Path
import mammoth

def generate_html_diff(original_docx: Path, processed_docx: Path, output_dir: Path):
    """
    Create a side-by-side HTML diff between original and processed DOCX files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert DOCX files to HTML using Mammoth
    with open(original_docx, "rb") as docx_file:
        result_orig = mammoth.convert_to_html(docx_file)
        html_orig = result_orig.value
    
    with open(processed_docx, "rb") as docx_file:
        result_proc = mammoth.convert_to_html(docx_file)
        html_proc = result_proc.value

    # Highlight new_text spans in yellow with CSS
    highlighted_proc = html_proc.replace(
        '<span>', '<span style="background-color:yellow;">'
    )

    html = f"""
    <html>
    <head>
        <style>
        .diff-container {{
            display: flex;
            flex-direction: row;
            gap: 20px;
        }}
        iframe {{
            width: 45vw;
            height: 90vh;
            border: 1px solid #444;
        }}
        </style>
    </head>
    <body>
    <h1>DOCX Diff: {original_docx.name} vs {processed_docx.name}</h1>
    <div class="diff-container">
        <iframe srcdoc='{html_orig.replace("'", "&apos;")}'></iframe>
        <iframe srcdoc='{highlighted_proc.replace("'", "&apos;")}'></iframe>
    </div>
    </body>
    </html>
    """
    out_path = output_dir / f"{original_docx.stem}_diff.html"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html)
    return out_path