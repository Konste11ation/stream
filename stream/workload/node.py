from abc import ABCMeta

from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.workload.layer_node_abc import LayerNodeABC


class Node(LayerNodeABC, metaclass=ABCMeta):
    """Abstract base class that represents a piece of an algorithmic workload.
    Example: ComputationNode, etc.
    """

    offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving]

    def __init__(  # noqa: PLR0913
        self,
        node_id: int,
        node_name: str,
        type: str,
        onchip_energy: float,
        offchip_energy: float,
        runtime: int,
        possible_core_allocation: list[int],
        chosen_core_allocation: int | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the Node metaclass

        Args:
            type: The type of Node.
            energy: The energy consumption of this Node.
            runtime: The runtime of this Node.
            possible_core_allocation: The core id on which this Node can be mapped.
            inputs: The names of the input tensors of this node
            outputs: The names of the output tensors of this node.
            chosen_core_allocation: The final core allocation of this node
            input_names: Names of the ONNX input node
        """
        if input_names is None:
            input_names = []
        super().__init__(node_id, node_name)

        self.type = type.lower()
        self.onchip_energy = onchip_energy
        self.offchip_energy = offchip_energy
        self.runtime = runtime
        self.possible_core_allocation = possible_core_allocation
        self.chosen_core_allocation = chosen_core_allocation
        self.input_names = input_names
        self.start = -1
        self.end = -1
        # number of data (in bits) only this node produces (not produced by any other node)
        self.data_produced_unique = 0
        # DVFS related
        self.dvfs_level: int = 0  # default DVFS level
        self.dvfs_mode: str = "Unset" # one of "DFS", "DVFS", "Unset", "Global"
        self.vdd_lut: dict[int, float] = {0: 1.0}  # default VDD LUT
        self.freq_lut: dict[int, float] = {0: 1.0}
        self.dyn_power_lut: dict[int, float] = {0: 1.0}
        self.sta_power_lut: dict[int, float] = {0: 1.0}
        self.absolute_static_power: float | None = None

    def get_total_energy(self) -> float:
        """Get the total energy of running this node, including off-chip energy."""
        return self.onchip_energy + self.offchip_energy

    def get_onchip_energy(self):
        """Get the on-chip energy of running this node.
        
        Energy = Dynamic Energy + Static Energy
        - Dynamic Energy scales with dyn_power_lut (representing V^2 drop or activity scaling)
        - Static Energy = Power * Time. 
          The user provides an absolute static power per core.
        """
        base_dyn_energy = self.onchip_energy
        
        if self.absolute_static_power is not None:
            # Power in mW is equivalent to pJ/ns.
            # Cycles to Time (ns) = cycles * (1000 / MHz)
            clock_mhz = getattr(self, 'system_clock_mhz', 1000.0)
            time_ns = self.runtime * (1000.0 / clock_mhz)
            base_sta_power = self.absolute_static_power * time_ns
        else:
            base_sta_power = 0.0
        
        if self.dvfs_level != 0:
            if self.dvfs_level in self.dyn_power_lut and self.dvfs_level in self.sta_power_lut:
                # 1. Scaling Factors
                dyn_factor = self.dyn_power_lut[self.dvfs_level]
                sta_factor = self.sta_power_lut[self.dvfs_level] # Represents Leakage Power scaling
                
                # 2. Runtime scaling
                freq_factor = self.freq_lut.get(self.dvfs_level, 1.0)
                time_scaling = 1.0 / freq_factor if freq_factor > 0 else 1.0
                
                # 3. Component Estimation
                scaled_dyn = base_dyn_energy * dyn_factor
                scaled_sta = base_sta_power * sta_factor * time_scaling
                
                return scaled_dyn + scaled_sta
                
        # Non-DVFS or missing LUT returns combined baseline
        return base_dyn_energy + base_sta_power

    def get_onchip_dynamic_energy(self):
        base_dyn_energy = self.onchip_energy
        if self.dvfs_level != 0 and self.dvfs_level in self.dyn_power_lut:
            return base_dyn_energy * self.dyn_power_lut[self.dvfs_level]
        return base_dyn_energy

    def get_onchip_static_energy(self):
        if self.absolute_static_power is not None:
            clock_mhz = getattr(self, 'system_clock_mhz', 1000.0)
            time_ns = self.runtime * (1000.0 / clock_mhz)
            base_sta_power = self.absolute_static_power * time_ns
        else:
            base_sta_power = 0.0

        if self.dvfs_level != 0 and self.dvfs_level in self.sta_power_lut:
            sta_factor = self.sta_power_lut[self.dvfs_level]
            freq_factor = self.freq_lut.get(self.dvfs_level, 1.0)
            time_scaling = 1.0 / freq_factor if freq_factor > 0 else 1.0
            return base_sta_power * sta_factor * time_scaling
            
        return base_sta_power

    def get_offchip_energy(self):
        """Get the off-chip energy of running this node."""
        return self.offchip_energy

    def get_runtime(self):
        """Get the runtime of running this node."""
        runtime = self.runtime
        if self.dvfs_level != 0:
            if self.dvfs_level in self.freq_lut:
                freq = self.freq_lut[self.dvfs_level]
                runtime_dvfs = int(runtime / freq)
                return runtime_dvfs
        else:
            return runtime

    def get_start(self):
        """Get the start time in cycles of this node."""
        return self.start

    def get_end(self):
        """Get the end time in cycles of this node."""
        return self.end

    def get_dvfs_level(self):
        """Get the DVFS level of this node."""
        return self.dvfs_level

    def set_onchip_energy(self, energy: float):
        """Set the on-chip energy of running this node.

        Args:
            energy (float): energy consumption of this node
        """
        self.onchip_energy = energy

    def set_offchip_energy(self, energy: float):
        """Set the off-chip energy of running this node.

        Args:
            energy (float): energy consumption of this node
        """
        self.offchip_energy = energy

    def set_runtime(self, runtime: int):
        """Set the runtime of running this node.

        Args:
            runtime: runtime in cycles
        """
        self.runtime = runtime

    def set_start(self, start: int):
        """Set the start time in cycles of this node.

        Args:
            start: start time in cycles
        """
        self.start = start

    def set_end(self, end: int):
        """Set the end time in cycles of this node.

        Args:
            end: end time in cycles
        """
        self.end = end

    def set_core_allocation(self, core_allocation: int):
        self.core_allocation = [core_allocation]

    def set_chosen_core_allocation(self, core_allocation: int | None):
        self.chosen_core_allocation = core_allocation

    def has_end(self) -> bool:
        """Check if this node has already been assigned an end time.

        Returns:
            bool: True if this node has been assigned an end time
        """
        return self.end is not None

    def set_offchip_bandwidth(self, offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving]):
        self.offchip_bandwidth_per_op = offchip_bandwidth_per_op

    def set_dvfs_level(self, dvfs_level: int):
        """Set the DVFS level for this node."""
        self.dvfs_level = dvfs_level

    def set_dvfs_mode(self, dvfs_mode: str):
        """Set the DVFS mode for this node. Expected 'DFS' or 'DVFS'."""
        self.dvfs_mode = dvfs_mode
        
    def set_vdd_lut(self, vdd_lut: dict[int, float]):
        """Set the VDD LUT for DVFS levels."""
        self.vdd_lut = vdd_lut
        
    def set_freq_lut(self, freq_lut: dict[int, float]):
        """Set the frequency LUT for DVFS levels."""
        self.freq_lut = freq_lut
        
    def set_dyn_power_lut(self, dyn_power_lut: dict[int, float]):
        """Set the dynamic energy LUT for DVFS levels."""
        self.dyn_power_lut = dyn_power_lut
        
    def set_sta_power_lut(self, sta_power_lut: dict[int, float]):
        """Set the static energy LUT for DVFS levels."""
        self.sta_power_lut = sta_power_lut

    def set_absolute_static_power(self, power: float):
        """Set the absolute static leakage power (per core)."""
        self.absolute_static_power = power

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
