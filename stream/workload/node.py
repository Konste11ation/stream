from abc import ABCMeta

from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.workload.layer_node_abc import LayerNodeABC

class Node(LayerNodeABC, metaclass=ABCMeta):
    """Abstract base class that represents a piece of an algorithmic workload.
    Example: ComputationNode, etc.
    """
    def __init__(
        self,
        node_id: int,
        node_name: str,
        type: str,
        onchip_energy: float,
        offchip_energy: float,
        runtime: int,
        possible_core_allocation: list[int],
        core_allocation_is_fixed: bool = False,
        chosen_core_allocation: int | None = None,
        # DVFS related parameters
        vdd_lut: dict[int, float] = {0:1.0},
        freq_lut: dict[int, float] = {0:1.0},
        energy_lut: dict[int, float] = {0:1.0},
    ) -> None:
        """Initialize the Node metaclass


        Args:
            type (str): The type of Node.
            energy (float): The energy consumption of this Node.
            runtime (int): The runtime of this Node.
            dvfs_level (int): The dvfs level of this node. Assume the higher the level, the lower the VDD. Default=0
            vdd_lut (dict[int, float]): The LUT to look the vdd scaling factor
            freq_lut (dict[int, float]): The LUT to look the freq scaling factor
            energy_lut (dict[int, float]): The LUT to look the energy scaling factor
            possible_core_allocation (int): The core id on which this Node can be mapped.
            inputs: (List[str]): The names of the input tensors of this node
            outputs: (List[str]): The names of the output tensors of this node.
            chosen_core_allocation: The final core allocation of this node
        """
        super().__init__(node_id, node_name)


        self.type = type.lower()
        self.onchip_energy = onchip_energy
        self.offchip_energy = offchip_energy
        self.runtime = runtime
        self.possible_core_allocation = possible_core_allocation
        self.core_allocation_is_fixed = core_allocation_is_fixed
        self.chosen_core_allocation = chosen_core_allocation
        # will be set by the scheduler
        self.start = None
        # will be set by the scheduler
        self.end = None
        # number of data (in bits) only this node consumes (not consumed by any other node)
        self.data_consumed_unique = 0
        # number of data (in bits) only this node produces (not produced by any other node)
        self.data_produced_unique = 0


        # will be set together with the core allocation
        self.offchip_bw = FourWayDataMoving(0, 0, 0, 0)
        # below are for DVFS
        self.vdd_lut = vdd_lut
        self.energy_lut = energy_lut
        self.freq_lut = freq_lut
        self.dvfs_levels: list[int] | None = None
        self.dvfs_window_duration: list[int] | None = None
        self.dvfs_runtime: int = 0
        self.dvfs_energy: float = 0.0
    # Getters
    def get_onchip_energy(self) -> float:
        """Get the on-chip energy of running this node."""
        return self.onchip_energy
    def get_onchip_dvfs_energy(self) -> float:
        """Get the on-chip dvfs energy of running this node."""
        return self.dvfs_energy
    def get_offchip_energy(self) -> float:
        """Get the off-chip energy of running this node."""
        return self.offchip_energy
    def get_total_energy(self) -> float:
        """Get the total energy of running this node, including off-chip energy."""
        return self.get_onchip_energy() + self.get_offchip_energy()
    def get_runtime(self):
        """Get the nominal runtime of running this node."""
        return self.runtime
    def get_dvfs_runtime(self):
        """Get the dvfs runtime of running this node."""
        return self.dvfs_runtime
    def get_vdd_lut(self):
        """Get the vdd LUT of running this node."""
        return self.vdd_lut
    def get_freq_lut(self):
        """Get the freq LUT of running this node."""
        return self.freq_lut
    def get_energy_lut(self):
        """Get the dynamic energy LUT of running this node."""
        return self.energy_lut
    def get_dvfs_levels(self):
        """Get the dvfs levels of running this node."""
        return self.dvfs_levels
    def get_start(self):
        """Get the start time in cycles of this node."""
        return self.start
    def get_end(self):
        """Get the end time in cycles of this node."""
        return self.end
    def get_chosen_core_allocation(self):
        return self.chosen_core_allocation
    
    # Setters
    def set_onchip_energy(self, energy: float):
        """Set the on-chip energy of running this node.
        Args:
            energy (float): energy consumption of this node
        """
        self.onchip_energy = energy
    def set_onchip_dvfs_energy(self, energy: float):
        """Set the on-chip dvfs energy of running this node.
        Args:
            energy (float): dvfs energy consumption of this node
        """
        self.dvfs_energy = energy
    def set_offchip_energy(self, energy: float):
        """Set the off-chip energy of running this node.
        Args:
            energy (float): energy consumption of this node
        """
        self.offchip_energy = energy


    def set_runtime(self, runtime: int):
        """Set the runtime of running this node.
        Args:
            runtime (int): runtime in cycles
        """
        self.runtime = runtime
    def set_dvfs_runtime(self, dvfs_runtime: int):
        """Set the dvfs runtime of running this node.
        Args:
            dvfs_runtime (int): dvfs runtime in cycles
        """
        self.dvfs_runtime = dvfs_runtime
    def set_dvfs_window_duration(self, dvfs_window_duration: list[int]):
        """Set the dvfs window duration of running this node.
        Args:
            dvfs_window_duration (list[int]): dvfs window duration in cycles
        """
        self.dvfs_window_duration = dvfs_window_duration
    def set_start(self, start: int):
        """Set the start time in cycles of this node.
        Args:
            start (int): start time in cycles
        """
        self.start = start


    def set_end(self, end: int):
        """Set the end time in cycles of this node.
        Args:
            end (int): end time in cycles
        """
        self.end = end
    
    def set_vdd_lut (self, vdd_lut:dict[int, float]):
        """Set the vdd LUT of this node."""
        self.vdd_lut = vdd_lut
    def set_freq_lut (self, freq_lut:dict[int, float]):
        """Set the freq LUT of this node."""
        self.freq_lut = freq_lut
    def set_energy_lut (self, energy_lut:dict[int, float]):
        """Set the dynamic energy LUT of this node."""
        self.energy_lut = energy_lut
    def set_core_allocation(self, core_allocation: int):
        self.core_allocation = [core_allocation]
    def set_chosen_core_allocation(self, core_allocation: int | None):
        self.chosen_core_allocation = core_allocation
    def set_dvfs_levels(self, dvfs_levels: list[int]):
        self.dvfs_levels = dvfs_levels


    def has_end(self) -> bool:
        """Check if this node has already been assigned an end time.


        Returns:
            bool: True if this node has been assigned an end time
        """
        return self.end is not None


    def set_offchip_bandwidth(self, offchip_bw: FourWayDataMoving):
        self.offchip_bw = offchip_bw


    def __str__(self):
        return self.name


    def __repr__(self):
        return self.nam