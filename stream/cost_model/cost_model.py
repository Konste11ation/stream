import networkx as nx
from zigzag.datatypes import LayerOperand
from stream.cost_model.scheduler import CoalaScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import plot_timeline_brokenaxes
from stream.workload.onnx_workload import ComputationNodeWorkload

class StreamCostModelEvaluation:
    """
    Evaluates the cost model for a given workload and accelerator using the Schedule class.
    Computes latency and various energy metrics by running the schedule.
    """

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        beam_width: int = 2,
    ) -> None:
        # Initialize the SCME by setting the workload graph to be scheduled
        self.workload = workload
        self.accelerator = accelerator
        self.energy: float | None = None
        self.total_cn_onchip_energy: float | None = None
        self.total_cn_offchip_link_energy: float | None = None
        self.total_cn_offchip_memory_energy: float | None = None
        self.total_eviction_to_offchip_link_energy: float | None = None
        self.total_eviction_to_offchip_memory_energy: float | None = None
        self.total_sink_layer_output_offchip_link_energy: float | None = None
        self.total_sink_layer_output_offchip_memory_energy: float | None = None
        self.total_core_to_core_link_energy: float | None = None
        self.total_core_to_core_memory_energy: float | None = None

        self.latency: int | None = None
        self.max_memory_usage = None
        self.core_timesteps_delta_cumsums = None
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.beam_width = beam_width

    def __str__(self):
        return f"SCME(energy={self.energy:.2e}, latency={self.latency:.2e})"

    def evaluate(self):
        """
        Runs the scheduling and cost model evaluation, updating latency and energy attributes.
        Uses the Schedule class for modular scheduling and result extraction.
        """
        # Run the scheduler directly
        schedule = CoalaScheduler(
            g=self.workload,
            accelerator=self.accelerator,
            scheduling_order=self.scheduling_order,
            operands_to_prefetch=self.operands_to_prefetch,
            beam_width=self.beam_width, # Configurable beam width for exploration
        )
        schedule.run()
        # Update the accelerator to the one used in the best schedule (since beam search creates copies)
        self.accelerator = schedule.accelerator
        schedule.update_graph_nodes()

        self.latency = schedule.latency
        self.total_cn_onchip_energy = schedule.total_cn_onchip_energy
        self.total_cn_offchip_link_energy = schedule.total_cn_offchip_link_energy
        self.total_cn_offchip_memory_energy = schedule.total_cn_offchip_memory_energy
        self.total_eviction_to_offchip_link_energy = schedule.total_eviction_to_offchip_link_energy
        self.total_eviction_to_offchip_memory_energy = schedule.total_eviction_to_offchip_memory_energy
        self.total_sink_layer_output_offchip_link_energy = schedule.total_sink_layer_output_offchip_link_energy
        self.total_sink_layer_output_offchip_memory_energy = schedule.total_sink_layer_output_offchip_memory_energy
        self.total_core_to_core_link_energy = schedule.total_core_to_core_link_energy
        self.total_core_to_core_memory_energy = schedule.total_core_to_core_memory_energy

        # Calculate idle leakage energy for the cores
        # For simplicity, approximate each core's baseline leakage power by taking the 
        # average leakage power of nodes executed on it. Idle time = latency - active_time.
        idle_energy = 0
        core_active_times = {core.id: 0 for core in self.accelerator.cores}
        core_leakage_powers = {core.id: [] for core in self.accelerator.cores}

        for node in schedule.scheduled_nodes:
            core_id = getattr(node, 'chosen_core_allocation', None)
            if core_id is None:
                core_id = node.core_allocation[0] if hasattr(node, 'core_allocation') and isinstance(node.core_allocation, list) else None
                
            if core_id is not None and core_id in core_active_times:
                rt = node.get_runtime() if node.get_runtime() else 0
                core_active_times[core_id] += rt
                if rt > 0:
                    # using the static ratio or absolute static power to compute base leakage
                    abs_sta = getattr(node, 'absolute_static_power', None)
                    if abs_sta is not None:
                        # cost_model leakage arrays operate natively in pJ per cycle natively elsewhere.
                        # We convert absolute static (mW = pJ/ns) to pJ/cycle.
                        # Time per cycle (ns) = 1000.0 / System Clock (MHz)
                        clock_mhz = getattr(node, 'system_clock_mhz', 1000.0)
                        base_leakage_power = abs_sta * (1000.0 / clock_mhz)
                    else:
                        base_leakage_power = 0.0
                    
                    sta_factor = 1.0
                    dvfs_lev = getattr(node, 'dvfs_level', 0)
                    if dvfs_lev != 0 and hasattr(node, 'sta_energy_lut'):
                        sta_factor = node.sta_energy_lut.get(dvfs_lev, 1.0)
                    
                    scaled_leakage_power = base_leakage_power * sta_factor
                    core_leakage_powers[core_id].append(scaled_leakage_power)

        for core in self.accelerator.cores:
            idle_time = max(0, self.latency - core_active_times[core.id])
            if core_leakage_powers[core.id]:
                avg_leakage = sum(core_leakage_powers[core.id]) / len(core_leakage_powers[core.id])
            else:
                avg_leakage = 0  # If no nodes scheduled, assume perfect power gating or 0 leakage
            idle_energy += idle_time * avg_leakage

        self.total_idle_energy = idle_energy

        self.energy = (
            self.total_cn_onchip_energy
            + self.total_idle_energy
            + self.total_cn_offchip_link_energy
            + self.total_cn_offchip_memory_energy
            + self.total_eviction_to_offchip_link_energy
            + self.total_eviction_to_offchip_memory_energy
            + self.total_sink_layer_output_offchip_link_energy
            + self.total_sink_layer_output_offchip_memory_energy
            + self.total_core_to_core_link_energy
            + self.total_core_to_core_memory_energy
        )

    def plot_schedule(
        self,
        plot_full_schedule: bool = False,
        draw_dependencies: bool = True,
        plot_data_transfer: bool = False,
        section_start_percent: tuple[int, ...] = (0, 50, 95),
        percent_shown: tuple[int, ...] = (5, 5, 5),
        fig_path: str = "outputs/schedule_plot.png",
    ):
        """Plot the schedule of this SCME."""
        if plot_full_schedule:
            section_start_percent = (0,)
            percent_shown = (100,)
        plot_timeline_brokenaxes(
            self,
            draw_dependencies,
            section_start_percent,
            percent_shown,
            plot_data_transfer,
            fig_path,
        )

    def plot_memory_usage(self, *args, **kwargs):
        """Plot the memory usage of this SCME."""
        plot_memory_usage(self, *args, **kwargs)
