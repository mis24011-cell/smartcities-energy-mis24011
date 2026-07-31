# -*- coding: utf-8 -*-
"""
Γενική συρραφή: αντικαθιστά ένα block-range του master document.xml με το σώμα
(body children) ενός παραγόμενου .docx, μεταφέροντας εικόνες + relationships.

Χρήση: splice(master_dir, source_docx, start_text, end_text_exclusive)
  - start_text: κείμενο του πρώτου block που ΑΝΤΙΚΑΘΙΣΤΑΤΑΙ (π.χ. η επικεφαλίδα)
  - end_text_exclusive: κείμενο του πρώτου block που ΔΙΑΤΗΡΕΙΤΑΙ μετά (όριο)
"""
import zipfile, re, os, shutil
from lxml import etree

W  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R  = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
RT = "http://schemas.openxmlformats.org/package/2006/relationships"
def qn(t): p,l=t.split(":"); return "{%s}%s"%({"w":W,"r":R}[p],l)

def block_text(el):
    return "".join(el.itertext()).strip()

def splice(master_dir, source_docx, start_text, end_text_excl, dry=False):
    doc_path = os.path.join(master_dir, "word", "document.xml")
    rels_path= os.path.join(master_dir, "word", "_rels", "document.xml.rels")
    media_dir= os.path.join(master_dir, "word", "media")

    tree = etree.parse(doc_path)
    root = tree.getroot()
    body = root.find(qn("w:body"))
    blocks = [c for c in body if c.tag in (qn("w:p"), qn("w:tbl"))]

    # εντοπισμός ορίων
    def find_idx(text, blocks):
        for i,b in enumerate(blocks):
            if block_text(b).startswith(text):
                return i
        return None
    si = find_idx(start_text, blocks)
    ei = find_idx(end_text_excl, blocks)
    assert si is not None, f"start not found: {start_text!r}"
    assert ei is not None, f"end not found: {end_text_excl!r}"
    assert si < ei, f"start({si}) must precede end({ei})"
    print(f"  range: blocks[{si}..{ei-1}] ({ei-si} blocks) → αντικαθίστανται")
    if dry:
        print("   first:", block_text(blocks[si])[:60])
        print("   last :", block_text(blocks[ei-1])[:60])
        print("   after:", block_text(blocks[ei])[:60])
        return

    # ── φόρτωση source docx: body children + rels + media ──
    zin = zipfile.ZipFile(source_docx)
    sdoc = etree.fromstring(zin.read("word/document.xml"))
    sbody= sdoc.find(qn("w:body"))
    try:
        srels_xml = zin.read("word/_rels/document.xml.rels")
        srels = etree.fromstring(srels_xml)
    except KeyError:
        srels = None

    # χάρτης rId(source) → media target
    src_media = {}
    if srels is not None:
        for rel in srels:
            if rel.get("Type","").endswith("/image"):
                src_media[rel.get("Id")] = rel.get("Target")  # π.χ. media/image1.png

    # master rels: μέγιστο rId + μέγιστο imageN
    mrels = etree.parse(rels_path).getroot()
    max_rid = max([int(m.group(1)) for r in mrels
                   for m in [re.match(r"rId(\d+)$", r.get("Id") or "")] if m] + [0])
    existing_media = set(os.listdir(media_dir))
    max_img = max([int(m.group(1)) for f in existing_media
                   for m in [re.match(r"image(\d+)\.", f)] if m] + [0])

    # μεταφορά κάθε source image → νέο media + νέο rel, χαρτογράφηση old_rid→new_rid
    rid_map = {}
    for old_rid, target in src_media.items():
        ext = os.path.splitext(target)[1]
        max_img += 1; max_rid += 1
        new_name = f"image{max_img}{ext}"
        with open(os.path.join(media_dir, new_name), "wb") as f:
            f.write(zin.read("word/"+target))
        new_rid = f"rId{max_rid}"
        rid_map[old_rid] = new_rid
        rel = etree.SubElement(mrels, "{%s}Relationship"%RT)
        rel.set("Id", new_rid)
        rel.set("Type","http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rel.set("Target", f"media/{new_name}")
    zin.close()

    # συλλογή νέων blocks (χωρίς sectPr) + remap r:embed
    new_blocks = []
    for c in sbody:
        if c.tag == qn("w:sectPr"):
            continue
        for blip in c.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            emb = blip.get(qn("r:embed"))
            if emb in rid_map:
                blip.set(qn("r:embed"), rid_map[emb])
        # remap και σε r:link αν υπάρχει (σπάνιο)
        new_blocks.append(c)

    # ── αντικατάσταση στο master body ──
    anchor = blocks[si]           # πρώτο προς αφαίρεση
    to_remove = blocks[si:ei]
    # εισαγωγή νέων ΠΡΙΝ το anchor
    idx_in_body = list(body).index(anchor)
    for off,nb in enumerate(new_blocks):
        body.insert(idx_in_body+off, nb)
    for b in to_remove:
        body.remove(b)

    # ── εγγραφή ──
    tree.write(doc_path, xml_declaration=True, encoding="UTF-8", standalone=True)
    etree.ElementTree(mrels).write(rels_path, xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"  ✓ εισήχθησαν {len(new_blocks)} blocks, {len(rid_map)} εικόνες")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    dry = "--dry" in args
    if dry: args.remove("--dry")
    master_dir, source_docx, start_text, end_text = args
    print(f"SPLICE {source_docx} → {start_text[:40]!r}..{end_text[:30]!r}")
    splice(master_dir, source_docx, start_text, end_text, dry=dry)
