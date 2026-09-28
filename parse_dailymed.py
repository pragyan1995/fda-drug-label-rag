import xml.etree.ElementTree as ET
import os
import glob

NS = {"hl7": "urn:hl7-org:v3"}

def get_text_content(text_elem):
    """Extract all readable text from a <text> element, ignoring styling tags."""
    if text_elem is None:
        return ""
    return " ".join(text_elem.itertext()).replace("\n", " ")
    # itertext() walks all nested tags and pulls their text, in order —
    # this naturally handles paragraph/list/item/content/br without
    # needing to special-case each tag.

def extract_sections(section_elem, results):
    """Recursively extract this section and any nested sub-sections."""
    code_elem = section_elem.find("hl7:code", NS)
    section_name = code_elem.get("displayName") if code_elem is not None else "UNKNOWN SECTION"

    # Skip the pure metadata/ingredient section — not clinical text
    if section_name and "SPL listing data elements" in section_name:
        pass
    else:
        text_elem = section_elem.find("hl7:text", NS)
        text_content = get_text_content(text_elem).strip()
        if text_content:
            results.append({"section": section_name, "text": text_content})

    # Recurse into nested sections (e.g. Warnings -> Do Not Use, Ask Doctor, etc.)
    for sub_component in section_elem.findall("hl7:component", NS):
        sub_section = sub_component.find("hl7:section", NS)
        if sub_section is not None:
            extract_sections(sub_section, results)

def parse_spl_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    results = []
    for component in root.findall(".//hl7:structuredBody/hl7:component", NS):
        section = component.find("hl7:section", NS)
        if section is not None:
            extract_sections(section, results)
    return results

def chunk_section_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks if chunks else [text]  # short sections stay as one chunk

# --- Run it on all pulled files ---
all_chunks = []
for filepath in glob.glob("dailymed_data/*.xml"):
    filename = os.path.basename(filepath)
    sections = parse_spl_file(filepath)
    for sec in sections:
        for chunk in chunk_section_text(sec["text"]):
            all_chunks.append({
                "source": filename,
                "section": sec["section"],
                "text": chunk
            })

print(f"Total chunks: {len(all_chunks)}")
for c in all_chunks[:5]:
    print(f"[{c['source']}] ({c['section']})")
    print(c["text"][:150], "...\n")