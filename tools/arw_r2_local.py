"""G1-R2 B2: small, independent RS1 quadrature on frozen sensor pixels.

The physical DOE phase is constant inside each DOE pixel.  Source boundary
pixels are integrated over the true disk intersection, split at every change
of the circular y-boundary.  This diagnostic never evaluates a full image.
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
from arw_reference.rs import propagate_rs1  # noqa: E402

PAIRS = [(2, 6), (4, 6), (6, 6), (6, 2), (6, 4)]
PIXELS = [0, 15, 31, 47, 63]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gauss_interval(order: int, left: float, right: float):
    x, w = np.polynomial.legendre.leggauss(order)
    return (left + right) / 2 + x * (right - left) / 2, w * (right - left) / 2


def disk_nodes(order: int, pitch: float, radius: float):
    """Return source nodes, integral weights, physical pixel ids, edge flags."""
    rows = []
    disk2 = radius * radius
    for iy in range(64):
        y0 = (iy - 32) * pitch
        y1 = y0 + pitch
        for ix in range(64):
            x0 = (ix - 32) * pitch
            x1 = x0 + pitch
            if min(abs(x0), abs(x1)) ** 2 + min(abs(y0), abs(y1)) ** 2 >= disk2 and not (x0 <= 0 <= x1 and y0 <= 0 <= y1):
                # The nearest point of this rectangle lies outside the disk.
                nx = 0 if x0 <= 0 <= x1 else min(abs(x0), abs(x1))
                ny = 0 if y0 <= 0 <= y1 else min(abs(y0), abs(y1))
                if nx * nx + ny * ny >= disk2:
                    continue
            interior = max(x0*x0, x1*x1) + max(y0*y0, y1*y1) <= disk2
            if interior:
                xg, wx = gauss_interval(order, x0, x1)
                yg, wy = gauss_interval(order, y0, y1)
                for x, a in zip(xg, wx):
                    for y, b in zip(yg, wy):
                        rows.append((x, y, a*b, iy*64+ix, False))
                continue
            # Add every x at which a circular boundary meets a rectangle edge.
            cuts = {x0, x1}
            for y in (y0, y1):
                if abs(y) < radius:
                    crossing = math.sqrt(disk2-y*y)
                    for x in (-crossing, crossing):
                        if x0 < x < x1:
                            cuts.add(x)
            for x in (-radius, radius):
                if x0 < x < x1:
                    cuts.add(x)
            ordered = sorted(cuts)
            for a, b in zip(ordered[:-1], ordered[1:]):
                xg, wx = gauss_interval(order, a, b)
                for x, vx in zip(xg, wx):
                    ycap = math.sqrt(max(0.0, disk2-x*x))
                    lo, hi = max(y0, -ycap), min(y1, ycap)
                    if hi <= lo:
                        continue
                    yg, wy = gauss_interval(order, lo, hi)
                    for y, vy in zip(yg, wy):
                        rows.append((x, y, vx*vy, iy*64+ix, True))
    data = np.asarray(rows, dtype=np.float64)
    return (data[:, :2], data[:, 2], data[:, 3].astype(np.int32), data[:, 4].astype(bool))


def sensor_nodes(order: int, indices: list[int], pitch: float):
    xbase, wbase = gauss_interval(order, 0, pitch)
    xy, weights, pixel = [], [], []
    for iy in indices:
        for ix in indices:
            for j, y in enumerate(xbase + (iy-32)*pitch):
                for i, x in enumerate(xbase + (ix-32)*pitch):
                    xy.append((x, y))
                    weights.append(wbase[i]*wbase[j])
                    pixel.append(iy*64+ix)
    return np.asarray(xy), np.asarray(weights), np.asarray(pixel)


def kernel(obs: np.ndarray, src: np.ndarray, wavelength: float, distance: float):
    delta = obs[:, None, :] - src[None, :, :]
    r = np.sqrt(np.sum(delta*delta, axis=-1) + distance*distance)
    k = 2*math.pi/wavelength
    return distance*np.exp(1j*k*r)/(1j*wavelength*r*r)*(1-1/(1j*k*r))


def checks(case: dict) -> dict:
    # A full rectangle has exact tensor-product area for every order.
    _, w = gauss_interval(6, -.007, .013)
    rectangle_error = abs(float(np.sum(w)**2) - .02**2)
    # A tiny direct sum checks the discrete FFT-RS indexing and source area.
    n, dx = 5, .002
    rng = np.random.default_rng(17)
    source = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    axis = (np.arange(n)-(n-1)/2)*dx
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    k = kernel(xy, xy, case["wavelength_mm"], case["propagation_mm"])
    direct = (k @ source.ravel()).reshape(n,n)*dx*dx
    fft = propagate_rs1(source.astype(np.complex128), dx, case["wavelength_mm"], case["propagation_mm"])
    direction = rng.normal(size=(n,n))
    tangent = (k @ (1j*direction*source).ravel()).reshape(n,n)*dx*dx
    h = .001
    plus = (k @ (source*np.exp(1j*h*direction)).ravel()).reshape(n,n)*dx*dx
    minus = (k @ (source*np.exp(-1j*h*direction)).ravel()).reshape(n,n)*dx*dx
    tangent_fd = (plus-minus)/(2*h)
    return {"rectangle_area_abs_error": rectangle_error,
            "tiny_direct_fft_relative_l2": float(np.linalg.norm(direct-fft)/np.linalg.norm(direct)),
            "tiny_direction_fd_relative_l2": float(np.linalg.norm(tangent-tangent_fd)/np.linalg.norm(tangent))}


def validate_protocol(full_protocol: dict):
    protocol = full_protocol["B2"]
    expected = {"case": "C2", "sensor_pixel_indices": PIXELS,
                "quadrature_pairs": [list(p) for p in PAIRS], "direction_count": 4,
                "dtype": "complex128"}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol {key} must equal {value!r}")
    if protocol.get("max_case_seconds", 0) > 600 or protocol.get("max_case_seconds", 0) <= 0:
        raise ValueError("case time cap must be in (0,600] seconds")
    if protocol.get("max_extra_ram_bytes", 0) > 1073741824 or protocol.get("max_extra_ram_bytes", 0) <= 0:
        raise ValueError("extra RAM cap must be in (0,1GiB]")
    if full_protocol.get("direction_seed") != 102:
        raise ValueError("C2 direction seed must be 102")
    if full_protocol.get("direction_generation") != "numpy.default_rng.standard_normal_sequential_rms":
        raise ValueError("direction generation must match original G1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    full_protocol = json.loads(protocol_path.read_text())
    validate_protocol(full_protocol)
    protocol = full_protocol["B2"]
    schedule_path = args.schedule.resolve()
    schedule = json.loads(schedule_path.read_text())
    if schedule.get("parent_protocol_sha256") != sha(protocol_path) or schedule.get("batch") != "B2" or schedule.get("pairs") != [list(p) for p in PAIRS] or schedule.get("sensor_pixel_indices") != PIXELS:
        raise ValueError("B2 schedule does not match the frozen protocol")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("B2 output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    cases_path = ROOT / "configs/week1/cases.json"
    case = next(c for c in json.loads(cases_path.read_text())["cases"] if c["id"] == "C2")
    phase_path = ROOT / case["initial_phase_file"]
    for rel in ("configs/week1/cases.json", case["initial_phase_file"]):
        if full_protocol["source_hashes"].get(rel) != sha(ROOT/rel):
            raise ValueError(f"frozen source hash changed: {rel}")
    phase = np.load(phase_path).astype(np.float64)
    rng = np.random.default_rng(102)
    directions = np.stack([v / np.sqrt(np.square(v).mean()) for v in
                           (rng.standard_normal((64,64)) for _ in range(4))])
    sensor = {q: sensor_nodes(q, PIXELS, case["sensor_pitch_mm"]) for q in (2,4,6)}
    sources = {q: disk_nodes(q, case["doe_pitch_mm"], case["aperture_radius_mm"]) for q in (2,4,6)}
    check = checks(case)
    check["disk_areas"] = {str(q): float(np.sum(sources[q][1])) for q in (2,4,6)}
    check["analytic_disk_area"] = math.pi*case["aperture_radius_mm"]**2
    check["disk_area_max_relative"] = max(abs(a-check["analytic_disk_area"])/check["analytic_disk_area"] for a in check["disk_areas"].values())
    if check["rectangle_area_abs_error"] > 1e-15 or check["tiny_direct_fft_relative_l2"] > 1e-11 or check["tiny_direction_fd_relative_l2"] > 1e-5 or check["disk_area_max_relative"] > 1e-4:
        raise RuntimeError(f"local quadrature correctness checks failed: {check}")
    # A representative block is timed before any full pair is evaluated.
    sq, tq = 6, 6
    sxy, sw, si, edge = sources[sq]
    oxy, ow, oi = sensor[tq]
    probe_obs = min(8, len(oxy))
    probe_src = min(1024, len(sxy))
    t0 = time.perf_counter()
    probe = kernel(oxy[:probe_obs], sxy[:probe_src], case["wavelength_mm"], case["propagation_mm"])
    _ = probe @ np.ones((probe_src,5),dtype=np.complex128)
    probe_time = time.perf_counter()-t0
    # Small pilots underestimate overhead; use a fixed 2x safety factor.
    pairs_count = sum(len(sources[s][0])*len(sensor[t][0]) for s,t in PAIRS)
    forecast = max(0.001, probe_time)*pairs_count/(probe_obs*probe_src)*protocol["pilot_time_safety_factor"]
    chunk_obs, chunk_src = 8, 1024
    # Kernel, coordinate deltas, radius and products, with 4x workspace margin.
    estimated_extra_ram = 4*chunk_obs*chunk_src*(16+16+8+16) + 128*sum(len(sources[q][0]) for q in sources)
    estimate = {"kernel_pairs": pairs_count, "pilot_pairs": probe_obs*probe_src,
                "pilot_seconds": probe_time, "forecast_seconds_safety2": forecast,
                "estimated_extra_ram_bytes": estimated_extra_ram,
                "source_nodes": {str(q): len(sources[q][0]) for q in sources},
                "sensor_nodes": {str(q): len(sensor[q][0]) for q in sensor},
                "boundary_source_nodes": {str(q): int(np.sum(sources[q][3])) for q in sources}}
    (output/"estimate.json").write_text(json.dumps({"protocol_sha256":sha(protocol_path), "schedule_sha256":sha(schedule_path), "estimate":estimate,"checks":check},indent=2)+"\n")
    if forecast > protocol["max_case_seconds"] or estimated_extra_ram > protocol["max_extra_ram_bytes"]:
        (output/"status.json").write_text(json.dumps({"status":"STOP_COST","estimate":estimate},indent=2)+"\n")
        print(json.dumps({"status":"STOP_COST","estimate":estimate}))
        return 4
    arrays = {"direction":directions}
    rows = []
    start = time.perf_counter()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for sq,tq in PAIRS:
        sxy, sw, si, edge = sources[sq]
        oxy, ow, oi = sensor[tq]
        input_features = np.concatenate((np.ones((1,len(si))),1j*directions.reshape(4,-1)[:,si]),axis=0)*np.exp(1j*phase.ravel()[si])[None,:]*sw[None,:]
        output_features = np.zeros((2,5,len(oxy)),dtype=np.complex128)
        # Final (6,6) pair accumulates symmetric phase perturbations with the same kernel.
        fd_features = None
        if (sq,tq) == (6,6):
            h = .001
            fd_features = np.stack([np.exp(1j*(phase.ravel()[si]+sign*h*directions[d].ravel()[si]))*sw
                                    for d in range(4) for sign in (1,-1)])
            fd_field = np.zeros((8,len(oxy)),dtype=np.complex128)
        for ostart in range(0,len(oxy),chunk_obs):
            oend = min(len(oxy),ostart+chunk_obs)
            for sstart in range(0,len(sxy),chunk_src):
                send = min(len(sxy),sstart+chunk_src)
                block = kernel(oxy[ostart:oend],sxy[sstart:send],case["wavelength_mm"],case["propagation_mm"])
                subset_edge = edge[sstart:send]
                for group, mask in enumerate((~subset_edge,subset_edge)):
                    if np.any(mask):
                        output_features[group,:,ostart:oend] += input_features[:,sstart:send][:,mask] @ block[:,mask].T
                if fd_features is not None:
                    fd_field[:,ostart:oend] += fd_features[:,sstart:send] @ block.T
            if time.perf_counter()-start > protocol["max_case_seconds"] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss_start > protocol["max_extra_ram_bytes"]:
                (output/"status.json").write_text(json.dumps({"status":"STOP_COST","reason":"actual time or memory cap reached; partial results omitted"},indent=2)+"\n")
                print(json.dumps({"status":"STOP_COST","output":str(output)}))
                return 4
        field = output_features.sum(axis=0)
        image_nodes = np.abs(field[0])**2
        dimage_nodes = 2*np.real(np.conj(field[0])[None,:]*field[1:])
        pixel_ids = np.array([y*64+x for y in PIXELS for x in PIXELS])
        image = np.array([np.sum(image_nodes[oi==p]*ow[oi==p])/case["sensor_pitch_mm"]**2 for p in pixel_ids])
        dimage = np.stack([np.array([np.sum(dimage_nodes[d,oi==p]*ow[oi==p])/case["sensor_pitch_mm"]**2 for p in pixel_ids]) for d in range(4)])
        label = f"s{sq}_t{tq}"
        arrays[f"{label}_source_interior"] = output_features[0]
        arrays[f"{label}_source_boundary"] = output_features[1]
        arrays[f"{label}_field"] = field
        arrays[f"{label}_sensor_weights"] = ow
        arrays[f"{label}_sensor_pixel_ids"] = oi
        arrays[f"{label}_intensity"] = image
        arrays[f"{label}_dintensity"] = dimage
        if fd_features is not None:
            fd_dimage = np.stack([np.array([np.sum((np.abs(fd_field[2*d,oi==p])**2-np.abs(fd_field[2*d+1,oi==p])**2)*ow[oi==p])/(2*h*case["sensor_pitch_mm"]**2) for p in pixel_ids]) for d in range(4)])
            check["final_fd_rel_l2"] = [float(np.linalg.norm(fd_dimage[d]-dimage[d])/max(np.linalg.norm(fd_dimage[d]),np.linalg.norm(dimage[d]),protocol["near_zero_absolute"])) for d in range(4)]
            arrays[f"{label}_fd_dintensity"] = fd_dimage
        rows.append({"source_order":sq,"sensor_order":tq,"source_nodes":len(sxy),"sensor_nodes":len(oxy),
                     "boundary_source_nodes":int(np.sum(edge)),"kernel_pairs":len(sxy)*len(oxy),
                     "mean_intensity_25":float(np.mean(image)),"mean_abs_dintensity_25":float(np.mean(np.abs(dimage)))})
    np.savez_compressed(output/"nodes_weights.npz",**{f"source{q}_{name}":arr for q,source in sources.items() for name,arr in zip(("xy","weight","pixel","boundary"),source)},**{f"sensor{q}_{name}":arr for q,obs in sensor.items() for name,arr in zip(("xy","weight","pixel"),obs)})
    np.savez_compressed(output/"local_raw.npz",**arrays)
    with (output/"local_quadrature.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparison = {}
    for a,b,name in [((2,6),(4,6),"source_2_4"),((4,6),(6,6),"source_4_6")]:
        for feature in ("field",):
            va=arrays[f"s{a[0]}_t{a[1]}_{feature}"]
            vb=arrays[f"s{b[0]}_t{b[1]}_{feature}"]
            comparison[name+"_field_rel_l2"]=[float(np.linalg.norm(va[d]-vb[d])/max(np.linalg.norm(va[d]),np.linalg.norm(vb[d]),protocol["near_zero_absolute"])) for d in range(5)]
        for group in ("source_interior","source_boundary"):
            va=arrays[f"s{a[0]}_t{a[1]}_{group}"]
            vb=arrays[f"s{b[0]}_t{b[1]}_{group}"]
            comparison[name+"_"+group+"_rel_l2"]=[float(np.linalg.norm(va[d]-vb[d])/max(np.linalg.norm(va[d]),np.linalg.norm(vb[d]),protocol["near_zero_absolute"])) for d in range(5)]
    for a,b,name in [((6,2),(6,4),"sensor_2_4"),((6,4),(6,6),"sensor_4_6")]:
        for feature in ("intensity","dintensity"):
            va=arrays[f"s{a[0]}_t{a[1]}_{feature}"]
            vb=arrays[f"s{b[0]}_t{b[1]}_{feature}"]
            if feature == "dintensity":
                comparison[name+"_"+feature+"_rel_l2"]=[float(np.linalg.norm(va[d]-vb[d])/max(np.linalg.norm(va[d]),np.linalg.norm(vb[d]),protocol["near_zero_absolute"])) for d in range(4)]
            else:
                comparison[name+"_"+feature+"_rel_l2"]=float(np.linalg.norm(va-vb)/max(np.linalg.norm(va),np.linalg.norm(vb),protocol["near_zero_absolute"]))
    if max(check["final_fd_rel_l2"]) > 1e-5:
        (output/"status.json").write_text(json.dumps({"status":"FAIL_CORRECTNESS","reason":"local derivative does not match its own finite difference","fd_relative_l2":check["final_fd_rel_l2"]},indent=2)+"\n")
        print(json.dumps({"status":"FAIL_CORRECTNESS","output":str(output)}))
        return 3
    result={"status":"SUCCESS","scope":"C2 25 fixed pixels only; not a G1 reference",
            "protocol_sha256":sha(protocol_path),"script_sha256":sha(Path(__file__)),
            "cases_sha256":sha(cases_path),"phase_sha256":sha(phase_path),
            "checks":check,"comparison":comparison,"estimate":estimate,
            "wall_seconds":time.perf_counter()-start,"peak_process_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    (output/"status.json").write_text(json.dumps({"status":"SUCCESS"},indent=2)+"\n")
    print(json.dumps({"status":"SUCCESS","output":str(output),"wall_seconds":result["wall_seconds"]}))


if __name__ == "__main__":
    sys.exit(main())
