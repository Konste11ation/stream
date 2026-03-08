import numpy as np
import yaml

def generate_dvfs_levels(v_base=1.0, v_th=0.3, alpha=1.5, num_levels=5, min_v=0.5):
    voltages = np.linspace(v_base, min_v, num_levels)
    
    levels = {}
    for i, v in enumerate(voltages):
        vdd = round(v, 2)
        
        # Frequency (Alpha Power Law)
        # f = K * (V - Vth)^alpha / V
        f_base = ((v_base - v_th)**alpha) / v_base
        f_curr = ((v - v_th)**alpha) / v
        freq = round(f_curr / f_base, 2)
        
        # Dynamic Energy (~ V^2)
        dyn_energy = round((v / v_base)**2, 2)
        
        # Static Power Scaling (P_leak proportional to V)
        # Assuming P_sta ~ V (simplified)
        p_base = v_base
        p_curr = v
        sta_power = round(p_curr / p_base, 2)
        
        levels[i] = [
            {"vdd": float(vdd)},
            {"freq": float(freq)},
            {"dyn_power": float(dyn_energy)},
            {"sta_power": float(sta_power)}
        ]
    return levels

if __name__ == "__main__":
    data = {"dvfs_level": generate_dvfs_levels()}
    print(yaml.dump(data, default_flow_style=False, sort_keys=False))
