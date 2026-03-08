import json
import os
import glob
import sys
from pathlib import Path

def analyze_scme_json(json_path):
    print(f"--- Analyzing {json_path} ---")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        traceEvents = data.get("traceEvents", [])
    elif isinstance(data, list):
        traceEvents = data
    else:
        traceEvents = []
    
    total_compute_energy = 0
    total_memory_energy = 0
    total_comm_energy = 0
    total_dynamic_energy = 0
    total_static_energy = 0
    
    core_active_times = {}
    tid_to_core = {}
    max_latency = 0
    
    # First pass to find core names
    for event in traceEvents:
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            tid = event.get("tid")
            core_name = event.get("args", {}).get("name", f"Unknown {tid}")
            tid_to_core[tid] = core_name

    for event in traceEvents:
        # Energy breakdown
        args = event.get("args", {})
        energy_val = args.get("Energy") or args.get("Energy (nJ)")
        if energy_val:
            e = float(energy_val)
            if "compute" in event.get("cat", "") or "Computation" in event["name"]:
                total_compute_energy += e
                if "EnergyDynamic" in args:
                    total_dynamic_energy += float(args["EnergyDynamic"])
                if "EnergyStatic" in args:
                    total_static_energy += float(args["EnergyStatic"])
            else:
                total_memory_energy += e # simplifying
                
        # Time breakdown
        if "ts" in event and "dur" in event:
            tid = event.get("tid", 0)
            core = tid_to_core.get(tid, f"Thread {tid}")
            dur = event["dur"]
            end_time = event["ts"] + dur
            if end_time > max_latency:
                max_latency = end_time
                
            if core not in core_active_times:
                core_active_times[core] = 0
            core_active_times[core] += dur
            
    total_energy = total_compute_energy + total_memory_energy + total_comm_energy
    
    print(f"Total Latency: {max_latency:.3e} cycles")
    print(f"Total Energy: {total_energy:.3e} nJ")
    if total_energy > 0:
        print(f"  -> Compute Energy: {total_compute_energy:.3e} nJ ({total_compute_energy/total_energy*100:.1f}%)")
        if total_dynamic_energy > 0 or total_static_energy > 0:
            print(f"     -> Dynamic: {total_dynamic_energy:.3e} nJ, Static: {total_static_energy:.3e} nJ")
        print(f"  -> Memory Energy: {total_memory_energy:.3e} nJ ({total_memory_energy/total_energy*100:.1f}%)")
    
    print("Core Utilization (Active Time / Total Latency):")
    for core, active_time in core_active_times.items():
        if isinstance(core, str) and ("Core" in core or "Accelerator" in core):
            util = (active_time / max_latency) * 100
            print(f"  -> {core}: {util:.1f}% (Idle: {100-util:.1f}%)")
            
    folder_name = os.path.basename(os.path.dirname(json_path))
    file_name = os.path.basename(json_path)
    name = f"{folder_name}/{file_name}" if folder_name else file_name
    
    result = {
        "path": json_path,
        "name": name,
        "latency": max_latency,
        "total_energy": total_energy,
        "dynamic_energy": total_dynamic_energy,
        "static_energy": total_static_energy,
        "compute_energy": total_compute_energy,
        "memory_energy": total_memory_energy,
        "core_util": {},
        "link_util": {}
    }
    for core, active_time in core_active_times.items():
        if isinstance(core, str):
            util = (active_time / max_latency) * 100
            if "->" in core:
                result["link_util"][core] = util
            elif "Core" in core or "Accelerator" in core:
                result["core_util"][core] = util
            
    return result

def print_comparison_summary(results):
    if not results:
        return
        
    print("\n" + "="*160)
    print("COMPARISON SUMMARY")
    print("="*160)
    
    # Find all unique cores across all runs to build columns
    all_cores = set()
    for r in results:
        all_cores.update(r["core_util"].keys())
    
    # Sort cores logically if they end in numbers
    def sort_key(x):
        parts = x.split()
        if parts and parts[-1].isdigit():
            return (0, int(parts[-1]))
        return (1, x)
        
    sorted_cores = sorted(list(all_cores), key=sort_key)
    
    # Build header
    core_headers = " | ".join([f"{c:^8}" for c in sorted_cores])
    header = f"{'Configuration':<35} | {'Latency':<12} | {'Energy(nJ)':<12} | {'Dyn(nJ)':<10} | {'Sta(nJ)':<10} | {core_headers} | {'GEMM Avg':^8} | {'GEMM STD':^8} | {'All Avg':^8} | {'Avg Link':^8}"
    print(header)
    print("-" * len(header))
    
    # Sort results: stage1 -> stage2 -> stage3 -> others
    def result_sort_key(r):
        name = r["name"].lower()
        if "stage1" in name: return 1
        if "stage2" in name: return 2
        if "stage3" in name: return 3
        return 4
        
    sorted_results = sorted(results, key=result_sort_key)
    
    # Filter to only show stages if we specify a folder, otherwise show all
    if any("stage" in r["name"].lower() for r in sorted_results):
        filtered_results = [r for r in sorted_results if "stage" in r["name"].lower()]
    else:
        filtered_results = sorted_results
    
    for r in filtered_results:
        # Use folder name as identifier
        name = r["name"]
        # If name is very long, truncate it
        if len(name) > 35:
            name = name[:32] + "..."
            
        cols = [
            f"{name:<35}",
            f"{r['latency']:<12.3e}",
            f"{r['total_energy']:<12.3e}",
            f"{r['dynamic_energy']:<10.3e}",
            f"{r['static_energy']:<10.3e}"
        ]
        for c in sorted_cores:
            util = r["core_util"].get(c, 0)
            cols.append(f"{util:>7.1f}%")
            
        # GEMM Avg and All Avg logic
        if sorted_cores:
            all_util = [r["core_util"].get(c, 0) for c in sorted_cores]
            all_avg = sum(all_util) / len(all_util)
            if len(all_util) > 1:
                gemm_utils = all_util[:-1]
                gemm_avg = sum(gemm_utils) / len(gemm_utils)
                gemm_std = (sum((x - gemm_avg)**2 for x in gemm_utils) / len(gemm_utils))**0.5
            else:
                gemm_avg = all_avg
                gemm_std = 0
        else:
            gemm_avg = 0
            gemm_std = 0
            all_avg = 0
            
        cols.append(f"{gemm_avg:>7.1f}%")
        cols.append(f"{gemm_std:>7.1f}%")
        cols.append(f"{all_avg:>7.1f}%")
        
        # Average Link Utilization
        avg_link = sum(r["link_util"].values()) / len(r["link_util"]) if r["link_util"] else 0
        cols.append(f"{avg_link:>7.1f}%")
        
        print(" | ".join(cols))
    print("="*len(header) + "\n")

if __name__ == "__main__":
    json_files = []
    if len(sys.argv) > 1:
        for target_path in sys.argv[1:]:
            if os.path.isfile(target_path) and target_path.endswith('.json'):
                json_files.append(target_path)
            else:
                # Assume it's a directory
                json_files.extend(glob.glob(f"{target_path}/**/*.json", recursive=True))
    else:
        # Default behavior
        outputs_dir = "outputs" 
        json_files = glob.glob(f"{outputs_dir}/**/*.json", recursive=True)
    
    if not json_files:
        print("No .json files found to analyze.")
        sys.exit(1)
        
    results = []
    for jf in json_files:
        res = analyze_scme_json(jf)
        if res:
            results.append(res)
            
    if len(results) >= 1:
        print_comparison_summary(results)
