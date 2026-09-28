"""Generate Agent check-input fixtures from real BMD Compute code (read-only)."""
import json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
compute_root = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path.insert(0, str(compute_root))
from pymatgen.core import Structure, Lattice
from backend.calculations.registry import desired_output_workflow_spec
from backend.calculations.default_treatments import resolve_default_treatments
from backend.calculations.input_reference import build_input_reference_payload
from backend.generated_inputs import generated_input_stage_previews
from backend.calculations.resources import normalize_execution_resources

def si(): return Structure(Lattice([[0,2.715,2.715],[2.715,0,2.715],[2.715,2.715,0]]), ["Si","Si"], [[0,0,0],[.25,.25,.25]])
def graphite(): return Structure(Lattice.hexagonal(2.464, 6.711), ["C"]*4, [[0,0,.25],[0,0,.75],[1/3,2/3,.25],[2/3,1/3,.75]])
rock = [[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5],[.5,.5,.5],[0,0,.5],[0,.5,0],[.5,0,0]]
def pbte(): return Structure(Lattice.cubic(6.46), ["Pb"]*4+["Te"]*4, rock)
def nio(): return Structure(Lattice.cubic(4.17), ["Ni"]*4+["O"]*4, rock)

def request(poscar, stage, theory, modifiers):
    # Exactly the request shape Agent's check-input sends.
    return {"structure": {"type": "pasted_text", "format": "poscar", "text": poscar},
            "workflow_spec": {"stages": [{"stage_type": stage, "theory": theory, "modifiers": list(modifiers), "label": None, "options": {}}],
                              "label": None, "recipe": None}}

cases, references = {}, []
def add_case(name, structure, desired, stage_index, reference_requests):
    resolution = resolve_default_treatments(structure, desired_output_workflow_spec(desired), desired_output=desired)
    preview = generated_input_stage_previews(structure, resolution.resolved_workflow,
        resources=normalize_execution_resources(None), potcar_functional="PBE_64")[stage_index - 1]
    stage = preview["stage_spec"]
    iset = preview["input_set"]
    poscar = str(iset.poscar)
    cases[name] = {
        "desired_output": desired,
        "stage_index": stage_index,
        "compute_stage": {"stage_type": stage.stage_type.value, "theory": stage.theory.value,
                          "modifiers": sorted(m.value for m in stage.modifiers)},
        "applied_treatments": [t.modifier.value for t in resolution.applied_treatments],
        "files": {"INCAR": str(iset.incar), "KPOINTS": str(iset.kpoints), "POSCAR": poscar},
    }
    for stage_type, theory, modifiers in reference_requests:
        req = request(poscar, stage_type, theory, modifiers)
        references.append({"case": name, "request": req,
                           "payload": build_input_reference_payload(req, repo_root=compute_root)})

add_case("si_pbe_static", si(), "energy_only", 1, [("static", "pbe", [])])
add_case("graphite_auto_d3", graphite(), "energy_only", 1, [("static", "pbe", [])])
add_case("pbte_auto_soc", pbte(), "energy_only", 1, [("static", "pbe", []), ("static", "pbe", ["soc"]), ("relax", "pbe", ["soc"]), ("static", "pbe", ["not_a_modifier"])])
add_case("nio_auto_spin", nio(), "energy_only", 1, [("static", "pbe", [])])
add_case("si_hse06_dos_stage", si(), "electronic_dos", 3, [("dos", "hse06", [])])
add_case("si_hse06_band_stage", si(), "electronic_band_structure", 3, [("band_structure", "hse06", [])])
out.write_text(json.dumps({"cases": cases, "references": references}, indent=1, sort_keys=True) + "\n")
print({r["case"] + ":" + ",".join(r["request"]["workflow_spec"]["stages"][0]["modifiers"]): r["payload"]["status"] for r in references})
print({c: v["applied_treatments"] for c, v in cases.items()})
print(references[0]["payload"]["producer"]["source"], references[0]["payload"]["request"].get("potcar_functional"))
for r in references:
    if r["payload"]["status"] != "ok": print(r["case"], r["payload"]["error"])
