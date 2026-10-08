"""G1-R2 B4: C1 three-grid RS check with the frozen boundary-area rule.

This is a conditional RS diagnostic, not the complete ASM/RS G1 gate.  It
does not modify the C2 B3 candidate or any historical G1/R1 result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.arw_r2_boundary import correctness_checks, disk_weights, sha  # noqa: E402
from tools.arw_r2_fine import ReusedRSKernel, detector, loss_and_direction, save_array  # noqa: E402


def frozen_inputs(protocol_path: Path, schedule_path: Path, output: Path):
    protocol=json.loads(protocol_path.read_text())
    schedule=json.loads(schedule_path.read_text())
    if not protocol.get("frozen_before_propagation") or schedule.get("parent_protocol_sha256") != sha(protocol_path):
        raise ValueError("B4 schedule does not bind frozen protocol")
    required={"batch":"B4","case":"C1","q":[4,6,8],"window_mm":3.2256,
              "directions":[0,1,2,3],"intervention":"boundary_area_fraction",
              "source_crop_detector_widths":2,"reference_b3_run":"r2-b3-01"}
    for key,value in required.items():
        if schedule.get(key)!=value:
            raise ValueError(f"B4 schedule changed {key}")
    if protocol.get("direction_generation")!="numpy.default_rng.standard_normal_sequential_rms":
        raise ValueError("direction generation changed")
    base=ROOT/"artifacts/G1-R2"
    if base not in output.parents or output.exists() and any(output.iterdir()):
        raise FileExistsError("B4 output must be a new empty G1-R2 subdirectory")
    b3=base/"runs/r2-b3-01"
    if b3 in output.parents or not (b3/"data/comparison_summary.json").is_file():
        raise ValueError("B4 cannot overwrite B3 and requires B3 evidence")
    b3_status=json.loads((b3/"status.json").read_text())
    b3_comparison=json.loads((b3/"data/comparison_summary.json").read_text())
    if b3_status["status"]!="SUCCESS" or not b3_comparison["continuation_conditions_all_met"]:
        raise ValueError("C2 B3 did not authorize C1 continuation")
    for rel,expected in schedule.get("prerequisite_sha256",{}).items():
        if sha(base/rel)!=expected:
            raise ValueError(f"B4 prerequisite changed: {rel}")
    return protocol,schedule


def sample_physical(phase: np.ndarray, directions: list[np.ndarray], n: int, dx: float, pitch: float):
    axis=(np.arange(n,dtype=np.float64)-(n-1)/2)*dx
    ix=np.floor(axis/pitch+32).astype(np.int64).clip(0,63)
    return phase[np.ix_(ix,ix)],[v[np.ix_(ix,ix)] for v in directions]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--schedule",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol_path,schedule_path,output=args.protocol.resolve(strict=True),args.schedule.resolve(strict=True),args.output.resolve()
    protocol,schedule=frozen_inputs(protocol_path,schedule_path,output)
    checks=correctness_checks()
    cases_path=ROOT/"configs/week1/cases.json"
    case=next(c for c in json.loads(cases_path.read_text())["cases"] if c["id"]=="C1")
    phase_path,target_path=ROOT/case["initial_phase_file"],ROOT/case["target_file"]
    for path in (cases_path,phase_path,target_path,ROOT/"arw_reference/rs.py"):
        rel=str(path.relative_to(ROOT))
        if protocol["source_hashes"][rel]!=sha(path):
            raise ValueError(f"frozen source changed: {rel}")
    phase=np.load(phase_path).astype(np.float64)
    target=np.load(target_path).astype(np.float64)
    if phase.shape!=(64,64) or target.shape!=(64,64):
        raise ValueError("C1 phase/target changed shape")
    rng=np.random.default_rng(101)
    directions=[]
    for _ in range(4):
        v=rng.standard_normal((64,64));v/=np.sqrt(np.square(v).mean());directions.append(v)
    output.mkdir(parents=True,exist_ok=True)
    raw=output/"raw_fields";raw.mkdir()
    hashes={}
    save_array(raw/"phase.npy",phase,hashes,output)
    save_array(raw/"target.npy",target,hashes,output)
    for d,v in enumerate(directions):
        save_array(raw/f"direction_{d}.npy",v,hashes,output)
    rows=[];settings=[]
    t0=time.perf_counter();cpu0=time.process_time()
    for q in schedule["q"]:
        ts=time.perf_counter();cs=time.process_time()
        dx=float(case["sensor_pitch_mm"])/q
        full_n=round(schedule["window_mm"]/dx)
        width=int(case["sensor_pixels"][0])*q
        n=schedule["source_crop_detector_widths"]*width
        crop_start=(full_n-n)//2
        if full_n<n or full_n-n!=2*crop_start or abs(full_n*dx-schedule["window_mm"])>1e-12:
            raise ValueError("C1 source support crop changes center or window")
        if (full_n-width)//2!=crop_start+(n-width)//2:
            raise ValueError("C1 detector indices change after source crop")
        alpha,area=disk_weights(n,dx,float(case["aperture_radius_mm"]))
        if area["area_relative_error"]>1e-10 or not area["interior_is_one"] or not area["outside_is_zero"]:
            raise ValueError("C1 aperture area check failed")
        if n*dx/2<=float(case["aperture_radius_mm"]):
            raise ValueError("C1 crop could omit nonzero source")
        sampled_phase,sampled_v=sample_physical(phase,directions,n,dx,float(case["doe_pitch_mm"]))
        source=(alpha*np.exp(1j*sampled_phase)).astype(np.complex128)
        input_power=float(alpha.sum()*dx*dx)
        squared_weight_power=float(np.vdot(source,source).real*dx*dx)
        if not input_power>squared_weight_power:
            raise ValueError("input power used squared boundary weights")
        qdir=raw/f"q{q}";qdir.mkdir()
        save_array(qdir/"alpha.npy",alpha,hashes,output)
        save_array(qdir/"source.npy",source,hashes,output)
        rs=ReusedRSKernel(n,dx,float(case["wavelength_mm"]),float(case["propagation_mm"]))
        observed=rs(source)
        save_array(qdir/"observed.npy",observed,hashes,output)
        base_image,_=detector(observed,np.zeros_like(observed),q,64)
        save_array(qdir/"image.npy",base_image,hashes,output)
        save_array(qdir/"normalized_image.npy",base_image/float(base_image.sum()),hashes,output)
        for d,v in enumerate(sampled_v):
            tangent=(1j*v*source).astype(np.complex128)
            source_power_derivative=float(2*np.real(np.vdot(source,tangent))*dx*dx)
            if abs(source_power_derivative)>1e-12*input_power:
                raise ValueError("C1 phase direction changed input power")
            save_array(qdir/f"source_direction_{d}.npy",tangent,hashes,output)
            observed_tangent=rs(tangent)
            save_array(qdir/f"observed_direction_{d}.npy",observed_tangent,hashes,output)
            image,dimage=detector(observed,observed_tangent,q,64)
            if not np.array_equal(image,base_image):
                raise ValueError("C1 direction changed base image")
            save_array(qdir/f"dimage_{d}.npy",dimage,hashes,output)
            parts,_,_=loss_and_direction(image,dimage,target,input_power,float(case["sensor_pitch_mm"]))
            rows.append({"case":"C1","q":q,"direction":d,"dx_mm":dx,
                         "full_n":full_n,"full_window_mm":schedule["window_mm"],
                         "source_crop_n":n,"source_crop_window_mm":n*dx,
                         "input_power":input_power,"squared_weight_input":squared_weight_power,
                         "input_power_derivative":source_power_derivative,**parts})
            del tangent,observed_tangent,dimage
        area.update({"q":q,"dx_mm":dx,"full_n":full_n,"source_crop_n":n,
                     "outside_source_power":0.0,"detector_indices_identical":True,
                     "kernel_build_count":1,"propagation_calls":rs.calls,
                     "wall_seconds":time.perf_counter()-ts,"cpu_seconds":time.process_time()-cs})
        settings.append(area)
        del alpha,source,observed,base_image,rs
    with (output/"c1_reference_check.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    indexed={(r["q"],r["direction"]):r for r in rows}
    adjacent=[]
    for a,b in ((4,6),(6,8)):
        im_a=np.load(raw/f"q{a}"/"normalized_image.npy")
        im_b=np.load(raw/f"q{b}"/"normalized_image.npy")
        power_a=indexed[(a,0)]["sensor_power"];power_b=indexed[(b,0)]["sensor_power"]
        directional=[]
        for d in range(4):
            da=indexed[(a,d)]["total_derivative"]
            db=indexed[(b,d)]["total_derivative"]
            amplitude=max(abs(da),abs(db))
            absolute=abs(da-db)
            near_zero=amplitude<protocol["acceptance"]["near_zero_amplitude"]
            relative=absolute/max(amplitude,1e-30)
            directional.append({"direction":d,"absolute_error":absolute,"relative_error":relative,
                                "near_zero_rule":near_zero,
                                "pass":bool(absolute<protocol["acceptance"]["near_zero_absolute"] if near_zero
                                            else relative<protocol["acceptance"]["direction_relative_lt"])})
        adjacent.append({"from_q":a,"to_q":b,
          "forward_normalized_l1":float(np.abs(im_a-im_b).sum()),
          "forward_power_relative":abs(power_a-power_b)/max(abs(power_a),abs(power_b),1e-30),
          "direction":directional})
    accept=protocol["acceptance"]
    all_grid=all(r["forward_normalized_l1"]<accept["forward_normalized_l1_lt"] and
                 r["forward_power_relative"]<accept["forward_power_relative_lt"] and
                 all(v["pass"] for v in r["direction"])
                 for r in adjacent)
    # A C1 25-pixel local check is a separate conditional run.  This records
    # only an estimate from the same-size C2 diagnostic; no local data is made.
    b2_summary=json.loads((ROOT/"artifacts/G1-R2/runs/r2-b2-01/data/summary.json").read_text())
    local_assessment={"needed_if_C1_grid_fails":not all_grid,
      "performed_in_B4":False,
      "C2_same_geometry_25pixel_measured_seconds":b2_summary["wall_seconds"],
      "C2_same_geometry_25pixel_peak_mib":b2_summary["peak_process_rss_mib"],
      "C1_forecast":"unknown until a C1 small-block timing pilot; C2 is only a scale comparison",
      "requires_new_frozen_schedule":True}
    (output/"local_need_assessment.json").write_text(json.dumps(local_assessment,indent=2)+"\n")
    comparison={"adjacent":adjacent,"all_grid_rules_met":all_grid,
                "local_assessment":local_assessment}
    (output/"c1_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    payload={"status":"SUCCESS","scope":"G1-R2 B4 C1 RS grid diagnostic; no ASM or gate decision",
      "protocol_sha256":sha(protocol_path),"schedule_sha256":sha(schedule_path),
      "script_sha256":sha(Path(__file__).resolve()),
      "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (cases_path,phase_path,target_path,ROOT/"arw_reference/rs.py",ROOT/"tools/arw_r2_boundary.py",ROOT/"tools/arw_r2_fine.py")},
      "raw_sha256":hashes,"correctness_checks":checks,"settings":settings,"comparison":comparison,
      "wall_seconds":time.perf_counter()-t0,"cpu_seconds":time.process_time()-cpu0,
      "peak_cpu_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
      "gpu_process_seconds":0,"gpu_kernel_seconds":0}
    (output/"c1_reference_check.json").write_text(json.dumps(payload,indent=2)+"\n")
    (output/"metrics.json").write_text(json.dumps({"status":"SUCCESS","batch":"B4",
       "wall_seconds":payload["wall_seconds"],"cpu_process_seconds":payload["cpu_seconds"],
       "peak_cpu_bytes":payload["peak_cpu_bytes"],"peak_gpu_bytes":0,
       "gpu_process_seconds":0,"gpu_kernel_seconds":0,"settings":3,"directions":12},indent=2)+"\n")
    print(json.dumps({"output":str(output),"rows":len(rows),"all_grid_rules_met":all_grid,
                      "wall_seconds":payload["wall_seconds"]}))


if __name__=="__main__":
    main()
