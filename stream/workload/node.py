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
        dvfs_level: int = 0,
        vdd_lut: dict[int, float] = {0:1.0},
        freq_lut: dict[int, float] = {0:1.0},
        energy_lut: dict[int, float] = {0:1.0},
        core_allocation_is_fixed: bool = False,
        chosen_core_allocation: int | None = None,
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
        self.dvfs_level = dvfs_level
        self.vdd_lut = vdd_lut
        self.energy_lut = energy_lut
        self.freq_lut = freq_lut
    def get_onchip_energy(self) -> float:
        """Get the on-chip energy of running this node."""
        energy_factor = self.energy_lut[self.dvfs_level]
        onchip_energy = self.onchip_energy*energy_factor
        return onchip_energy
    def get_offchip_energy(self) -> float:
        """Get the off-chip energy of running this node."""
        return self.offchip_energy

    def get_total_energy(self) -> float:
        """Get the total energy of running this node, including off-chip energy."""
        return self.get_onchip_energy() + self.get_offchip_energy()
    def get_runtime(self):
        """Get the runtime of running this node."""
        freq_factor = self.freq_lut[self.dvfs_level]
        runtime = int(self.runtime/freq_factor)
        return runtime
    def get_vdd(self):
         """Get the vdd of running this node."""
         vdd = self.vdd_lut[self.dvfs_level]
         return vdd
    def get_start(self):
        """Get the start time in cycles of this node."""
        return self.start

    def get_end(self):
        """Get the end time in cycles of this node."""
        return self.end

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
            runtime (int): runtime in cycles
        """
        self.runtime = runtime

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
        self.vdd_lut = vdd_lut
    def set_freq_lut (self, freq_lut:dict[int, float]):
        self.freq_lut = freq_lut
    def set_energy_lut (self, energy_lut:dict[int, float]):
        self.energy_lut = energy_lut 
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

    def set_offchip_bandwidth(self, offchip_bw: FourWayDataMoving):
        self.offchip_bw = offchip_bw

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
