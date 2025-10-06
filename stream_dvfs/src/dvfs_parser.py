import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_DIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_DIR))
import logging
from zigzag.utils import open_yaml
logger = logging.getLogger(__name__)

class DvfsParser:
    def __init__(self, dvfs_yaml_path: str):
        self.dvfs_yaml_path = dvfs_yaml_path
    def run(self):
        dvfs_luts = self.parse_dvfs_data()
        return dvfs_luts

    def parse_dvfs_data(self) -> dict[str,dict[int,float]]:
        dvfs_data = open_yaml(self.dvfs_yaml_path)
        dvfs_levels = dvfs_data['dvfs_level']
        vdd_lut = {}
        freq_lut = {}
        dyn_energy_lut = {}
        sta_energy_lut = {}
        for level, entries in dvfs_levels.items():
            combined = {}
            for entry in entries:
                combined.update(entry)
            vdd_lut[level] = combined['vdd']
            freq_lut[level] = combined['freq']
            dyn_energy_lut[level] = combined['dyn_energy']
            sta_energy_lut[level] = combined['sta_energy']
        dvfs_luts = {
            'vdd_lut': vdd_lut,
            'freq_lut': freq_lut,
            'dyn_energy_lut': dyn_energy_lut,
            'sta_energy_lut': sta_energy_lut,
        }
        return dvfs_luts