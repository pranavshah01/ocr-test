import os
import zipfile
import shutil
from lxml import etree

# WordprocessingML namespace
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def duplicate_side_by_side_sections(docx_filename, output_filename):
    # Get script location so paths are always relative to this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(BASE_DIR, docx_filename)
    output_path = os.path.join(BASE_DIR, output_filename)

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_dir = os.path.join(BASE_DIR, "tmp_docx")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # Extract DOCX contents (DOCX is just a ZIP)
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    doc_xml_path = os.path.join(tmp_dir, "word", "document.xml")
    tree = etree.parse(doc_xml_path)

    # ===== Pass 1: Duplicate table rows with two side-by-side cells =====
    for table in tree.xpath("//w:tbl", namespaces=NS):
        for tr in table.xpath("./w:tr", namespaces=NS):
            cells = tr.xpath("./w:tc", namespaces=NS)
            if len(cells) == 2:
                cell_has_content = False
                for cell in cells:
                    xml_str = etree.tostring(cell, encoding="unicode")
                    if ("<w:t" in xml_str) or ("<w:drawing" in xml_str) or ("<w:pict" in xml_str):
                        cell_has_content = True
                if cell_has_content:
                    tr_copy = etree.fromstring(etree.tostring(tr))
                    tr.addnext(tr_copy)

    # ===== Pass 2: Duplicate paragraphs with multiple images/shapes/textboxes =====
    for p in tree.xpath("//w:p", namespaces=NS):
        drawings = p.xpath(".//w:drawing | .//w:pict", namespaces=NS)
        if len(drawings) >= 2:  # multiple drawings in one paragraph
            p_copy = etree.fromstring(etree.tostring(p))
            p.addnext(p_copy)

    # Save modified document.xml
    tree.write(doc_xml_path, xml_declaration=True, encoding="UTF-8", standalone="yes")

    # Repackage into a new DOCX
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as new_docx:
        for foldername, subfolders, filenames in os.walk(tmp_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, tmp_dir)
                new_docx.write(file_path, arcname)

    # Clean up
    shutil.rmtree(tmp_dir)

    print(f"✅ Duplicated sections saved to: {output_path}")


# ===== Example Usage =====
if __name__ == "__main__":
    # These can be relative to the script's folder
    duplicate_side_by_side_sections(
        docx_filename="source_documents/testimages_092051.docx",       # source file (relative to script)
        output_filename="processed/testimages_092051_duplicated.docx"  # output file (relative to script)
    )