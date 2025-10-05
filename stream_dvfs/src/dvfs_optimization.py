import sys
import os
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent
STREAM_DIR = STREAM_DVFS_DIR.parent
sys.path.append(str(STREAM_DIR))
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.hardware.architecture.accelerator import Accelerator
import logging
logger = logging.getLogger(__name__)
from collections import defaultdict
class DvfsOptimizationStage(Stage):

    def __init__(self, 
                 list_of_callables: list[StageCallable],
                *,
                workload: ComputationNodeWorkload,
                accelerator: Accelerator,
                scheduling_order: list[tuple[int, int]],
                **kwargs
                ):
        super().__init__(list_of_callables, **kwargs)
        self.dvfs_output_fig_path = kwargs["dvfs_output_path"]
        self.workload = workload
        self.accelerator = accelerator
        self.scheduling_order = scheduling_order
        self.ga_nb_generations = kwargs["ga_nb_generations"]
        self.ga_nb_individuals = kwargs["ga_nb_individuals"]
    
    def get_energy(self):
        energy = sum(n.get_onchip_energy() for n in self.workload.node_list)
        return energy
    def get_latency(self):
        latency = max(n.end for n in self.workload.node_list)
        return latency
    def get_communication_dic(self):
        """
        Return all the output transfer event as a dic
        key:(id,sub_id)
        value:{start, end, runtime, tensors}
        """
        active_links = self.accelerator.communication_manager.get_all_links()
        node_events = {}
        for pair_link_id, cl in enumerate(active_links):
            for event in cl.events:
                start = event.start
                end = event.end
                runtime = end - start
                tensor = event.tensor
                node = event.tensor.origin
                tensor_type =  event.tensor.memory_operand
                node_id = node.id
                node_sub_id = node.sub_id

                if runtime == 0:
                    continue
                if not tensor_type.is_output():
                    continue
                key = (node_id,node_sub_id)
                event_record = {
                    "Start": start,
                    "End": end,
                    "Runtime": runtime,
                    "Tensors": tensor
                }
                node_events.setdefault(key, event_record)
        return node_events
    def get_start_time_per_core(self):
        """
        Retuen a dict to store the start_time per core per node
        key: core
        value: [(node, start_time), ...]
        """
        start_time_per_core = defaultdict(list)
        
        for node in self.workload.node_list:
            core = node.chosen_core_allocation
            start_time = node.get_start()
            start_time_per_core[core].append((node, start_time))
        # sorted by the start time
        # smallest first
        for core in start_time_per_core:
            start_time_per_core[core].sort(key=lambda x: x[1])
        return start_time_per_core
    def find_next_start_time_per_core(self,start_time_per_core, core, end_time):
        # Check if the core exist
        if core not in start_time_per_core:
            return float("nan")
        
        # Get all the nodes running on the current core
        core_nodes = start_time_per_core[core]

        # find the nxt start time after the current end time
        for _,start_time in core_nodes:
            if(start_time>=end_time):
                return start_time
        return float("nan")
    def run(self):
        logger.info(f"Start DVFS optimization stage")
        base_energy = self.get_energy()
        base_latency = self.get_latency()        
        print(f"Base Energy = {base_energy}, Base Latency = {base_latency}")
        self.brute_dvfs_opt()
    def brute_dvfs_opt(self):
        node_event_dic = self.get_communication_dic()
        start_time_per_core = self.get_start_time_per_core()
        for node in self.workload.node_list:
            cur_id = node.id
            cur_sub_id = node.sub_id
            cur_end = node.get_end()
            cur_runtime = node.get_runtime()
            cur_core = node.chosen_core_allocation
            successor_nodes = set(self.workload.successors(node))
            successor_nodes_start_times = [n.start for n in successor_nodes]
            # the current node is the exit node
            if successor_nodes_start_times == []:
                continue
            # get the output transfer time
            output_transfer_event = node_event_dic.get((cur_id, cur_sub_id), [])
            output_transfer_time = output_transfer_event["Runtime"] if output_transfer_event else 0 
            # the earlist start time of the successor
            est_successors = min(successor_nodes_start_times)
            # the earlist start time of the current core node
            est_core =  self.find_next_start_time_per_core(start_time_per_core,
                                                           cur_core,
                                                           cur_end)
            deadline = min(est_successors-output_transfer_time,est_core)
            slack = deadline - cur_end
            if slack>0:
                print(f"Node {node} has a slack of {slack}, cur runtime {cur_runtime}")
        
    def is_leaf(self):
        return True