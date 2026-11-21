"""Builds catalogue/templates.json and catalogue/crosswalk.json (single source of truth for template geometry).
Coordinates: x from left, y from top, as fractions of the frame. Mapping to PICD / CADB labels is by NAME (draft, not validated)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
PHI = (3 - 5 ** 0.5) / 2  # 0.381966
T = []
def add(id, name, family, kind, prims, overlay, picd, cadb, status, note=""):
    T.append(dict(id=id, name=name, family=family, kind=kind, primitives=prims, overlay_names=overlay, picd=picd, cadb=cadb, status=status, note=note))

# ---- v0: position templates
add("thirds_h", "Rule of thirds, horizontal lines", "thirds", "hline", [1/3, 2/3], ["Rule of Thirds"], ["LS-Hori3"], ["rule of thirds", "horizontal"], "v0")
add("thirds_v", "Rule of thirds, vertical lines", "thirds", "vline", [1/3, 2/3], ["Rule of Thirds"], ["L-Ver3"], ["rule of thirds", "vertical"], "v0")
add("thirds_pts", "Rule of thirds, power points", "thirds", "points", [[a, b] for a in (1/3, 2/3) for b in (1/3, 2/3)], ["Rule of Thirds"], ["P-RoT", "S-RoT"], ["rule of thirds"], "v0")
add("centre_h", "Centre, horizontal midline", "centre", "hline", [0.5], ["Centre Cross", "Quadrant"], ["LS-Hori2"], ["center", "horizontal"], "v0", "Centre Cross and Quadrant have the same geometry (the two midlines); the page lists them separately.")
add("centre_v", "Centre, vertical midline", "centre", "vline", [0.5], ["Centre Cross", "Quadrant"], ["L-Ver2"], ["center", "vertical", "symmetric"], "v0")
add("centre_pt", "Centre point", "centre", "points", [[0.5, 0.5]], ["Centre Cross", "Quadrant"], ["P-Cent", "S-Cent"], ["center"], "v0")
add("phi_h", "Golden-ratio grid, horizontal lines", "phi", "hline", [PHI, 1 - PHI], ["Golden Ratio", "Phi (Golden)"], [], ["golden ratio"], "v0", "No PICD category is defined by phi positions; 0.382 lies 4.9% of the frame from 1/3.")
add("phi_v", "Golden-ratio grid, vertical lines", "phi", "vline", [PHI, 1 - PHI], ["Golden Ratio", "Phi (Golden)"], [], ["golden ratio"], "v0")
add("phi_pts", "Golden-ratio grid, power points", "phi", "points", [[a, b] for a in (PHI, 1 - PHI) for b in (PHI, 1 - PHI)], ["Golden Ratio"], [], ["golden ratio"], "v0")
add("fifths_h", "Rule of fifths, horizontal lines", "fifths", "hline", [0.2, 0.4, 0.6, 0.8], ["Rule of Fifths"], [], [], "v0")
add("fifths_v", "Rule of fifths, vertical lines", "fifths", "vline", [0.2, 0.4, 0.6, 0.8], ["Rule of Fifths"], [], [], "v0")
add("diag_baroque", "Baroque diagonal (bottom-left to top-right)", "diagonal", "diag", ["BL-TR"], ["Diagonal Method", "Baroque Diagonal"], ["LS-Dia", "P-Dia"], ["diagonal"], "v0", "Corner-to-corner line in pixel geometry (angle depends on the aspect ratio).")
add("diag_sinister", "Sinister diagonal (top-left to bottom-right)", "diagonal", "diag", ["TL-BR"], ["Diagonal Method", "Sinister Diagonal"], ["LS-Dia", "P-Dia"], ["diagonal"], "v0")

# ---- deferred (reason recorded)
add("golden_triangle", "Golden triangle", "diagonal", "composite", [], ["Golden Triangle", "Reciprocal Lines"], [], ["triangle"], "M2b", "Diagonal plus two perpendiculars from the other corners. Reciprocal lines appear to share this geometry; confirm against a primary source.")
add("rabatment", "Rabatment", "frame", "vline", [], ["Rabatment"], [], [], "M2b", "Lines at the edges of the squares on the short side; depends on the aspect ratio.")
add("golden_spiral", "Golden spiral", "curve", "path", [], ["Golden Spiral"], [], ["curved"], "M2b", "Eight orientations (four rotations, two mirrors); needs a curve evidence map.")
add("persp_1pt", "One-point perspective", "perspective", "vp", [], ["1-Point"], ["S-Per"], ["vanishing point"], "M2b", "Needs vanishing-point evidence (LSD + RANSAC; the benchmark's B1 block is a starting point).")
add("persp_2pt", "Two-point perspective", "perspective", "vp", [], ["2-Point"], ["S-Per"], ["vanishing point"], "M2b")
add("persp_3pt", "Three-point perspective", "perspective", "vp", [], ["3-Point"], ["S-Per"], ["vanishing point"], "M2b")
add("armature14", "Armature, 14-line", "armature", "composite", [], ["Armature, 14-line"], [], [], "deferred", "Geometry not pinned down in the sources read; needs a primary source before it is implemented.")
add("rootrect", "Root rectangles and phi rectangle (frame proportion)", "frame", "frame", [], ["Root 2", "Root 3", "Root 4", "Root 5", "Phi (Golden)", "1.5 Rectangle"], [], [], "frame-only", "A property of the frame's aspect ratio, not of image content; reported as a descriptor.")
add("radial", "Radial / circular", "curve", "path", [], ["Radial / Circular"], [], ["radial"], "deferred")
add("symmetry", "Mirror symmetry", "symmetry", "mirror", [], [], [], ["symmetric"], "deferred", "Not in the overlay catalogue; present in CADB and likely derivable from ISR symmetry readings.")

json.dump(T, open(os.path.join(HERE, "templates.json"), "w"), indent=1)

# ---- crosswalk
PICD = {1:"P-RoT",2:"S-RoT",3:"P-Cent",4:"S-Cent",5:"P-Dia",6:"LS-Dia",7:"P-Hori",8:"LS-Hori2",9:"LS-Hori3",10:"S-Hori",11:"P-Ver",12:"L-Ver2",13:"L-Ver3",14:"L-VerMul",15:"P-Tri",16:"LS-Tri",17:"LS-C-Cur",18:"LS-O-Cur",19:"LS-S-Cur",20:"LS-Dif",21:"S-Per",22:"PL-Den",23:"PL-Pat",24:"P-Scat"}
CADB = ["center","rule of thirds","golden ratio","triangle","horizontal","vertical","diagonal","symmetric","curved","radial","vanishing point","pattern","fill the frame"]
cw = {"note": "Draft mapping by NAME, made 20 Nov 2026; not validated. A PICD category can map to several templates (it is a label, not a rule).",
      "picd": {abbr: [t["id"] for t in T if abbr in t["picd"]] for abbr in PICD.values()},
      "cadb": {c: [t["id"] for t in T if c in t["cadb"]] for c in CADB},
      "templates_without_picd": [t["id"] for t in T if not t["picd"]],
      "picd_without_v0_template": [a for a, ids in ((a, [t["id"] for t in T if a in t["picd"] and t["status"] == "v0"]) for a in PICD.values()) if not ids]}
json.dump(cw, open(os.path.join(HERE, "crosswalk.json"), "w"), indent=1)
v0 = [t for t in T if t["status"] == "v0"]
print(len(T), "templates;", len(v0), "in v0;", "PICD categories with a v0 template:", 24 - len(cw["picd_without_v0_template"]), "of 24")
print("PICD without v0 template:", cw["picd_without_v0_template"])
print("CADB classes without any template:", [c for c, v in cw["cadb"].items() if not v])
