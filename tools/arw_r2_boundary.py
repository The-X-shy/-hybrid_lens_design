"""G1-R2 B3: one C2 boundary-area candidate on the existing sampled RS kernel.

The real disk/square intersection area weights the *field* linearly.  Input
power uses the same area fraction linearly, not the squared field amplitude.
This is a conditional diagnostic and does not itself qualify G1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.arw_r2_fine import ReusedRSKernel, detector, loss_and_direction, save_array  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_primitive(x: float, r: float) -> float:
    x = max(-r, min(r, x))
    return 0.5 * (x * math.sqrt(max(0.0, r*r-x*x)) + r*r*math.asin(x/r))


def disk_square_area(x0: float, x1: float, y0: float, y1: float, r: float) -> float:
    """Area of an axis-aligned rectangle intersecting the disk, analytically."""
    left, right = max(x0, -r), min(x1, r)
    if right <= left:
        return 0.0
    cuts = {left, right}
    for y in (y0, y1):
        if abs(y) < r:
            hit = math.sqrt(max(0.0, r*r-y*y))
            for x in (-hit, hit):
                if left < x < right:
                    cuts.add(x)
    area = 0.0
    ordered = sorted(cuts)
    for a, b in zip(ordered[:-1], ordered[1:]):
        mid = (a+b)/2
        cap = math.sqrt(max(0.0, r*r-mid*mid))
        upper_cap = cap <= y1
        lower_cap = -cap >= y0
        upper = cap if upper_cap else y1
        lower = -cap if lower_cap else y0
        if upper <= lower:
            continue
        constant = (0.0 if upper_cap else y1) - (0.0 if lower_cap else y0)
        arc_count = int(upper_cap) + int(lower_cap)
        area += constant*(b-a) + arc_count*(circular_primitive(b,r)-circular_primitive(a,r))
    return max(0.0, min((x1-x0)*(y1-y0), area))


def disk_weights(n: int, dx: float, r: float):
    """Full sampled grid; exact disk area fraction only on straddling cells."""
    edges = (np.arange(n+1, dtype=np.float64)-n/2)*dx
    x0, x1 = edges[:-1], edges[1:]
    active = np.flatnonzero((x0 < r) & (x1 > -r))
    a0, a1 = x0[active], x1[active]
    closest = np.where(a0 > 0, a0, np.where(a1 < 0, -a1, 0.0))
    farthest = np.maximum(np.abs(a0), np.abs(a1))
    nearest_sq = np.square(closest)[:,None] + np.square(closest)[None,:]
    farthest_sq = np.square(farthest)[:,None] + np.square(farthest)[None,:]
    interior = farthest_sq <= r*r
    outside = nearest_sq >= r*r
    alpha = np.zeros((n,n), dtype=np.float64)
    alpha[np.ix_(active,active)] = interior.astype(np.float64)
    boundary = np.argwhere(~interior & ~outside)
    for iy, ix in boundary:
        gy, gx = active[iy], active[ix]
        alpha[gy,gx] = disk_square_area(x0[gx],x1[gx],x0[gy],x1[gy],r)/(dx*dx)
    if np.min(alpha) < -1e-12 or np.max(alpha) > 1+1e-12:
        raise ValueError("invalid circle-cell area fraction")
    return alpha, {"fully_inside_count": int(interior.sum()),
                   "boundary_count": int(len(boundary)),
                   "outside_count": int(n*n-interior.sum()-len(boundary)),
                   "area_mm2": float(alpha.sum()*dx*dx),
                   "analytic_area_mm2": math.pi*r*r,
                   "area_relative_error": float(abs(alpha.sum()*dx*dx-math.pi*r*r)/(math.pi*r*r)),
                   "interior_is_one": bool(np.all(alpha[np.ix_(active,active)][interior] == 1)),
                   "outside_is_zero": bool(np.all(alpha[np.ix_(active,active)][outside] == 0))}


def sample_phase_directions(phase: np.ndarray, directions: np.ndarray, n: int, dx: float, pitch: float):
    axis = (np.arange(n, dtype=np.float64)-(n-1)/2)*dx
    iy = np.floor(axis/pitch+32).astype(np.int64)
    ix = iy.copy()
    # All nonzero alpha cells fit well inside the 64 x 64 physical DOE.
    cy, cx = iy.clip(0,63), ix.clip(0,63)
    return phase[np.ix_(cy,cx)], [v[np.ix_(cy,cx)] for v in directions]


def frozen_inputs(protocol_path: Path, schedule_path: Path, output: Path):
    protocol = json.loads(protocol_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    if protocol.get("frozen_before_propagation") is not True:
        raise ValueError("unfrozen parent protocol")
    if schedule.get("parent_protocol_sha256") != sha(protocol_path) or schedule.get("batch") != "B3":
        raise ValueError("B3 schedule does not bind the frozen parent protocol")
    b3 = protocol.get("B3", {})
    # B3 is frozen as a child schedule; the parent may omit the conditional plan.
    if schedule.get("case") != "C2" or schedule.get("intervention") != "boundary_area_fraction":
        raise ValueError("B3 may contain only the frozen C2 boundary-area candidate")
    if schedule.get("q") != [8,12,16] or float(schedule.get("window_mm",0)) != 1.6128:
        raise ValueError("B3 grid differs from B1")
    if schedule.get("directions") != [0,1,2,3]:
        raise ValueError("B3 changed directions or sensor integration")
    if schedule.get("reference_b1_run") != "r2-b1-01" or schedule.get("reference_b2_run") != "r2-b2-01":
        raise ValueError("B3 comparison runs changed")
    if (schedule.get("source_rule") != "alpha is exact intersection area of the subgrid cell with hard circular aperture divided by cell area; same phase and direction per physical DOE pixel"
        or schedule.get("propagation_field_weight") != "alpha*exp(i*phase)"
        or schedule.get("physical_input_power") != "sum(alpha)*dx_mm^2, never sum(alpha^2)*dx_mm^2"
        or schedule.get("sensor_rule") != "unchanged mean of intensity within original 64x64 physical sensor pixels"):
        raise ValueError("B3 source, power, or sensor convention changed")
    if protocol.get("direction_seed") != 102:
        raise ValueError("direction seed changed")
    if b3 and b3.get("candidate") != "boundary_area_fraction":
        raise ValueError("parent B3 candidate changed")
    base = ROOT / "artifacts/G1-R2"
    if output == base or not base in output.parents or output.exists() and any(output.iterdir()):
        raise FileExistsError("B3 output must be a new empty G1-R2 subdirectory")
    if (base/"runs/r2-b1-01") in output.parents or (base/"runs/r2-b2-01") in output.parents:
        raise ValueError("B3 cannot overwrite B1 or B2")
    b1 = base / "runs/r2-b1-01/data"
    b2 = base / "runs/r2-b2-01/data"
    if not (b1/"fine_baseline.csv").is_file() or not (b2/"local_raw.npz").is_file():
        raise FileNotFoundError("B1 and B2 evidence required")
    for batch in ("b1","b2"):
        status = json.loads((base/f"runs/r2-{batch}-01/status.json").read_text())
        if status["status"] != "SUCCESS":
            raise ValueError(f"{batch} was not successful")
    for rel, expected in schedule.get("decision_evidence_sha256", {}).items():
        if sha(base/rel) != expected:
            raise ValueError(f"B3 decision evidence changed: {rel}")
    return protocol, schedule, b1, b2


def correctness_checks():
    r = 1.0
    checks = {
        "full_square": disk_square_area(-.1,.1,-.1,.1,r),
        "empty_square": disk_square_area(1.1,1.2,0,.1,r),
        "quarter_disk": disk_square_area(0,1,0,1,r),
        "circle_area": disk_square_area(-1,1,-1,1,r),
    }
    if (abs(checks["full_square"]-.04)>1e-14 or checks["empty_square"] != 0
        or abs(checks["quarter_disk"]-math.pi/4)>1e-13
        or abs(checks["circle_area"]-math.pi)>1e-12):
        raise RuntimeError(f"disk/square integration check failed: {checks}")
    n, dx, wl, z = 5, .002, .0007, 10.0
    rng=np.random.default_rng(17)
    source=(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))).astype(np.complex128)
    axis=(np.arange(n)-(n-1)/2)*dx
    yy,xx=np.meshgrid(axis,axis,indexing="ij")
    xy=np.column_stack((xx.ravel(),yy.ravel()))
    offset=xy[:,None,:]-xy[None,:,:]
    radius=np.sqrt(np.sum(offset*offset,axis=-1)+z*z)
    k=2*math.pi/wl
    kernel=z*np.exp(1j*k*radius)/(1j*wl*radius*radius)*(1-1/(1j*k*radius))
    direct=(kernel@source.ravel()).reshape(n,n)*dx*dx
    fft=ReusedRSKernel(n,dx,wl,z)(source)
    checks["tiny_direct_fft_relative_l2"]=float(np.linalg.norm(direct-fft)/np.linalg.norm(direct))
    if checks["tiny_direct_fft_relative_l2"]>1e-11:
        raise RuntimeError("tiny direct sum and FFT convolution differ")
    checks["fraction_power_identity"] = "P_in=sum(alpha)*dx^2; field input=sum(alpha^2)*dx^2 separately reported"
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path, schedule_path, output = args.protocol.resolve(strict=True), args.schedule.resolve(strict=True), args.output.resolve()
    protocol, schedule, b1_dir, b2_dir = frozen_inputs(protocol_path,schedule_path,output)
    checks = correctness_checks()
    case_path = ROOT/"configs/week1/cases.json"
    case = next(c for c in json.loads(case_path.read_text())["cases"] if c["id"] == "C2")
    phase_path, target_path = ROOT/case["initial_phase_file"], ROOT/case["target_file"]
    for path in (case_path,phase_path,target_path,ROOT/"arw_reference/rs.py"):
        rel = str(path.relative_to(ROOT))
        if protocol["source_hashes"][rel] != sha(path):
            raise ValueError(f"frozen source changed: {rel}")
    phase = np.load(phase_path).astype(np.float64)
    target = np.load(target_path).astype(np.float64)
    directions = np.stack([np.load(b1_dir/"raw_fields"/f"direction_{i}.npy") for i in range(4)])
    rng=np.random.default_rng(102)
    for v in directions:
        expected=rng.standard_normal((64,64));expected/=np.sqrt(np.square(expected).mean())
        if not np.array_equal(v,expected):
            raise ValueError("B1 direction differs from original G1")
    with (b1_dir/"fine_baseline.csv").open() as handle:
        b1_rows={(int(r["q"]),int(r["direction"])):r for r in csv.DictReader(handle)}
    b2=np.load(b2_dir/"local_raw.npz")
    fixed_indices=[0,15,31,47,63]
    fixed_ix=np.ix_(fixed_indices,fixed_indices)
    output.mkdir(parents=True,exist_ok=True)
    raw=output/"raw_fields"
    raw.mkdir()
    hashes={}
    rows=[]
    local_rows=[]
    areas=[]
    t0=time.perf_counter();cpu0=time.process_time()
    for q in schedule["q"]:
        ts=time.perf_counter();cs=time.process_time()
        dx=float(case["sensor_pitch_mm"])/q
        full_n=round(schedule["window_mm"]/dx)
        n=2*64*q
        if full_n<n or (full_n-n)%2 or abs(full_n*dx-schedule["window_mm"])>1e-12:
            raise ValueError("invalid full-window/source-crop index")
        alpha,area=disk_weights(n,dx,float(case["aperture_radius_mm"]))
        if area["area_relative_error"]>1e-10 or not area["interior_is_one"] or not area["outside_is_zero"]:
            raise ValueError(f"disk area check failed q={q}: {area}")
        area.update({"q":q,"dx_mm":dx,"n":n})
        areas.append(area)
        qdir=raw/f"q{q}";qdir.mkdir()
        save_array(qdir/"alpha.npy",alpha,hashes,output)
        sampled_phase,sampled_v=sample_phase_directions(phase,directions,n,dx,float(case["doe_pitch_mm"]))
        # The original phase/direction remain piecewise constant per DOE pixel.
        field=(alpha*np.exp(1j*sampled_phase)).astype(np.complex128)
        old_source=np.load(b1_dir/"raw_fields"/f"q{q}"/"source.npy")
        interior_mask=alpha==1.0
        interior_max_difference=float(np.max(np.abs(field[interior_mask]-old_source[interior_mask])))
        if interior_max_difference>1e-14:
            raise ValueError("B3 changed original fully interior source cells")
        area["interior_source_max_abs_difference_to_B1"]=interior_max_difference
        del old_source,interior_mask
        physical_input_power=float(alpha.sum()*dx*dx)
        squared_weight_input=float(np.vdot(field,field).real*dx*dx)
        if not physical_input_power>squared_weight_input or abs(physical_input_power-area["area_mm2"])>1e-14:
            raise ValueError("physical input power used squared area weight")
        save_array(qdir/"source.npy",field,hashes,output)
        rs=ReusedRSKernel(n,dx,float(case["wavelength_mm"]),float(case["propagation_mm"]))
        observed=rs(field)
        save_array(qdir/"observed.npy",observed,hashes,output)
        zero=np.zeros_like(observed)
        base_image,_=detector(observed,zero,q,64)
        del zero
        save_array(qdir/"image.npy",base_image,hashes,output)
        for direction_id,sv in enumerate(sampled_v):
            tangent=(1j*sv*field).astype(np.complex128)
            save_array(qdir/f"source_direction_{direction_id}.npy",tangent,hashes,output)
            dfield=rs(tangent)
            save_array(qdir/f"observed_direction_{direction_id}.npy",dfield,hashes,output)
            image,dimage=detector(observed,dfield,q,64)
            save_array(qdir/f"dimage_{direction_id}.npy",dimage,hashes,output)
            parts,_,_=loss_and_direction(image,dimage,target,physical_input_power,float(case["sensor_pitch_mm"]))
            original=b1_rows[(q,direction_id)]
            row={"case":"C2","q":q,"direction":direction_id,
                 "dx_mm":dx,"full_n":full_n,"source_crop_n":n,
                 "input_power":physical_input_power,"squared_weight_input":squared_weight_input,
                 "input_power_analytic":math.pi*float(case["aperture_radius_mm"])**2,
                 **parts}
            for key in ("sensor_power","sensor_power_derivative","image_loss","power_loss",
                        "total_loss","image_derivative","power_derivative","total_derivative"):
                row[f"B1_{key}"]=float(original[key])
                row[f"delta_{key}"]=row[key]-row[f"B1_{key}"]
            rows.append(row)
            if q==16:
                b2_image=b2["s6_t6_intensity"].reshape(5,5)
                b2_dimage=b2["s6_t6_dintensity"][direction_id].reshape(5,5)
                b1_image=np.load(b1_dir/"raw_fields/q16/image.npy")[fixed_ix]
                b1_dimage=np.load(b1_dir/"raw_fields/q16"/f"dimage_{direction_id}.npy")[fixed_ix]
                for iy,py in enumerate(fixed_indices):
                    for ix,px in enumerate(fixed_indices):
                        local_rows.append({"q":q,"direction":direction_id,"sensor_y":py,"sensor_x":px,
                                           "B1_intensity":float(b1_image[iy,ix]),"B3_intensity":float(image[py,px]),
                                           "B2_intensity":float(b2_image[iy,ix]),
                                           "B1_dintensity":float(b1_dimage[iy,ix]),
                                           "B3_dintensity":float(dimage[py,px]),
                                           "B2_dintensity":float(b2_dimage[iy,ix])})
            del tangent,dfield,dimage
        area["propagation_calls"]=rs.calls
        area["wall_seconds"]=time.perf_counter()-ts
        area["cpu_seconds"]=time.process_time()-cs
        del alpha,field,observed,base_image,rs

    for name,data in (("intervention.csv",rows),("local_comparison.csv",local_rows)):
        with (output/name).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    by_setting={(int(row["q"]),int(row["direction"])):row for row in rows}
    local_summary=[]
    for d in range(4):
        selected=[r for r in local_rows if r["direction"]==d]
        ref=np.array([r["B2_dintensity"] for r in selected])
        old=np.array([r["B1_dintensity"] for r in selected])
        new=np.array([r["B3_dintensity"] for r in selected])
        scale=max(float(np.linalg.norm(ref)),1e-30)
        local_summary.append({"direction":d,
                              "B1_vs_B2_dintensity_rel_l2":float(np.linalg.norm(old-ref)/scale),
                              "B3_vs_B2_dintensity_rel_l2":float(np.linalg.norm(new-ref)/scale)})
    adjacent=[]
    for a,b in ((8,12),(12,16)):
        first=np.load(raw/f"q{a}"/"image.npy")
        second=np.load(raw/f"q{b}"/"image.npy")
        first=first/first.sum();second=second/second.sum()
        power_a=by_setting[(a,0)]["sensor_power"]
        power_b=by_setting[(b,0)]["sensor_power"]
        adjacent.append({"from_q":a,"to_q":b,
                         "forward_normalized_l1":float(np.abs(first-second).sum()),
                         "forward_power_relative":abs(power_a-power_b)/max(abs(power_a),abs(power_b),1e-30),
                         "direction_relative":[abs(by_setting[(a,d)]["total_derivative"]-by_setting[(b,d)]["total_derivative"])
                           /max(abs(by_setting[(a,d)]["total_derivative"]),abs(by_setting[(b,d)]["total_derivative"]),1e-30)
                           for d in range(4)]})
    improvement=schedule["candidate_improvement_rule"]
    continuation=schedule["continue_to_C1_and_formal_only_if"]
    direction0_last=adjacent[-1]["direction_relative"][0]
    local_max=max(r["B3_vs_B2_dintensity_rel_l2"] for r in local_summary)
    comparison={"adjacent":adjacent,"local_25pixel":local_summary,
      "candidate_improves_both_measures":bool(
        direction0_last<improvement["direction0_last_grid_relative_below_old"] and
        local_max<improvement["local_25pixel_dintensity_max_below_old"]),
      "continuation_conditions_all_met":bool(
        all(r["forward_normalized_l1"]<continuation["both_adjacent_forward_l1_lt"] and
            r["forward_power_relative"]<continuation["both_adjacent_power_relative_lt"] and
            all(v<continuation["both_adjacent_all4_direction_relative_lt"] for v in r["direction_relative"])
            for r in adjacent) and local_max<continuation["local_25pixel_dintensity_max_lt"]),
      "direction0_last_grid_relative":direction0_last,
      "local_25pixel_dintensity_rel_l2_max":local_max}
    (output/"comparison_summary.json").write_text(json.dumps(comparison,indent=2)+"\n")
    payload={"status":"SUCCESS","scope":"one C2 boundary source-area candidate; no sensor change; not G1 gate",
             "protocol_sha256":sha(protocol_path),"schedule_sha256":sha(schedule_path),
             "script_sha256":sha(Path(__file__).resolve()),
             "b1_baseline_csv_sha256":sha(b1_dir/"fine_baseline.csv"),
             "b2_local_raw_sha256":sha(b2_dir/"local_raw.npz"),
             "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (case_path,phase_path,target_path,ROOT/"arw_reference/rs.py",ROOT/"tools/arw_r2_fine.py")},
             "raw_sha256":hashes,"correctness_checks":checks,"areas":areas,
             "comparison":comparison,
             "wall_seconds":time.perf_counter()-t0,"cpu_seconds":time.process_time()-cpu0,
             "peak_cpu_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
             "gpu_process_seconds":0,"gpu_kernel_seconds":0}
    (output/"intervention.json").write_text(json.dumps(payload,indent=2)+"\n")
    (output/"metrics.json").write_text(json.dumps({"status":"SUCCESS","batch":"B3",
       "wall_seconds":payload["wall_seconds"],"cpu_process_seconds":payload["cpu_seconds"],
       "peak_cpu_bytes":payload["peak_cpu_bytes"],"peak_gpu_bytes":0,
       "gpu_process_seconds":0,"gpu_kernel_seconds":0,"settings":3,"directions":12},indent=2)+"\n")
    print(json.dumps({"output":str(output),"rows":len(rows),"local_rows":len(local_rows),
                      "wall_seconds":payload["wall_seconds"]}))


if __name__ == "__main__":
    main()
