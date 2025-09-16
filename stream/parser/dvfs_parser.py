from typing import Any

from zigzag.utils import open_yaml

class DvfsParser:
    def __init__(self, dvfs_yaml_path: str):
        self.dvfs_yaml_path = dvfs_yaml_path
    def run(self):
        dvfs_luts = self.parse_dvfs_data()
        return dvfs_luts


    def parse_dvfs_data(self) -> dict[str,dict[int,float]]:
        dvfs_data = open_yaml(self.dvfs_yaml_path)
        min_dvfs_switch_latency = dvfs_data.get('min_dvfs_switch_latency', 1.0)
        system_clock_freq = dvfs_data.get('system_clock_freq', 1.0)
        dvfs_levels = dvfs_data['dvfs_level']
        vdd_lut = {}
        freq_lut = {}
        energy_lut = {}
        for level, entries in dvfs_levels.items():
            combined = {}
            for entry in entries:
                combined.update(entry)
            vdd_lut[level] = combined['vdd']
            freq_lut[level] = combined['freq']
            energy_lut[level] = combined['energy']
        dvfs_luts = {
            'vdd_lut': vdd_lut,
            'freq_lut': freq_lut,
            'energy_lut': energy_lut,
            'min_dvfs_switch_latency': min_dvfs_switch_latency,
            'system_clock_freq': system_clock_freq
        }
        return dvfs_luts