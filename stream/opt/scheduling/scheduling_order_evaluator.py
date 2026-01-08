import logging
import time
import os
import gc
from typing import List, Tuple, Optional, Set
from zigzag.utils import pickle_deepcopy
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

from stream.cost_model.scheduler import CoalaScheduler
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.hardware.architecture.accelerator import Accelerator
from zigzag.datatypes import LayerOperand

logger = logging.getLogger(__name__)

def get_total_energy(scheduler: CoalaScheduler) -> float:
    return (
        scheduler.total_cn_onchip_energy
        + scheduler.total_cn_offchip_link_energy
        + scheduler.total_cn_offchip_memory_energy
        + scheduler.total_eviction_to_offchip_link_energy
        + scheduler.total_eviction_to_offchip_memory_energy
        + scheduler.total_sink_layer_output_offchip_link_energy
        + scheduler.total_sink_layer_output_offchip_memory_energy
        + scheduler.total_core_to_core_link_energy
        + scheduler.total_core_to_core_memory_energy
    )

def evaluate_order_job(order: List[Tuple[int, int]], base_scheduler: CoalaScheduler) -> Tuple[float, float, CoalaScheduler]:
    """
    Worker function to evaluate a single order.
    """
    # Disable GC during heavy lifting to speed up allocation
    gc.disable()
    try:
        t0 = time.time()
        # new_scheduler = pickle_deepcopy(base_scheduler)
        # Use fast copy method if available, else fallback
        if hasattr(base_scheduler, 'copy'):
             new_scheduler = base_scheduler.copy()
        else:
             new_scheduler = pickle_deepcopy(base_scheduler)

        t_copy = time.time() - t0
        
        # We need to update the scheduling order for the new scheduler to pick the correct next node
        new_scheduler.scheduling_order = order
        new_scheduler._initialize_scheduling_order_lookup()
        
        # Schedule the next node
        t1 = time.time()
        new_scheduler.schedule_next_node()
        t_schedule = time.time() - t1
        
        latency = new_scheduler.get_total_latency()
        energy = get_total_energy(new_scheduler)
        # logger.info(f"Copy time: {t_copy*1000:.2f}ms, Schedule time: {t_schedule*1000:.2f}ms")
    finally:
        gc.enable()
    
    return latency, energy, new_scheduler

def evaluate_batch_job(base_scheduler: CoalaScheduler, orders: List[List[Tuple[int, int]]]) -> List[Tuple[float, float, CoalaScheduler]]:
    """
    Worker function to evaluate a batch of orders derived from the same base scheduler.
    This reduces the overhead of pickling/unpickling the base_scheduler.
    """
    results = []
    # We pickle_deepcopy the base_scheduler once per batch if we want to be safe, 
    # OR we pickle_deepcopy it for every item. 
    # Since we modify it, we MUST copy it for every item.
    # But passing 'base_scheduler' to this function only happens once via IPC.
    
    for order in orders:
        results.append(evaluate_order_job(order, base_scheduler))
    return results

class SchedulingOrderEvaluator:
    """
    Evaluates different scheduling orders using a Beam Search approach with CoalaScheduler as the cost function.
    """
    def __init__(self, workload: ComputationNodeWorkload, accelerator: Accelerator, operands_to_prefetch: Optional[List[LayerOperand]] = None):
        self.workload = workload
        self.accelerator = accelerator
        self.operands_to_prefetch = operands_to_prefetch

    def evaluate_partial_order_incremental(self, order: List[Tuple[int, int]], base_scheduler: CoalaScheduler) -> Tuple[float, float, CoalaScheduler]:
        """
        Evaluates a partial order incrementally starting from a base scheduler state.
        Returns (latency, energy, new_scheduler_state).
        """
        # Delegate to helper
        return evaluate_order_job(order, base_scheduler)

    def evaluate_baseline_order(self) -> Tuple[float, float]:
        """
        Evaluates the baseline topological order using CoalaScheduler.
        Returns (latency, energy).
        """
        baseline_order = [(n.id, n.sub_id) for n in nx.lexicographical_topological_sort(self.workload)]
        
        scheduler = CoalaScheduler(
            g=pickle_deepcopy(self.workload),
            accelerator=pickle_deepcopy(self.accelerator),
            scheduling_order=baseline_order,
            operands_to_prefetch=self.operands_to_prefetch
        )
        
        scheduler.run()
        
        latency = scheduler.latency
        energy = get_total_energy(scheduler)
        
        return latency, energy

    def run(self, beam_width: int = 3, max_time: int = 600, parallel: bool = False, max_workers: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Performs a Beam Search to find an optimal topological order incrementally.
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        logger.info(f"Starting Beam Search with width {beam_width}, max_time {max_time}s, parallel={parallel}, workers={max_workers}")
        start_time = time.time()
        
        # Calculate baseline
        base_latency, base_energy = self.evaluate_baseline_order()
        
        # Initial State: Empty order, initial scheduler state
        # Create an initial scheduler that hasn't scheduled anything yet
        init_scheduler = CoalaScheduler(
            g=pickle_deepcopy(self.workload),
            accelerator=pickle_deepcopy(self.accelerator),
            scheduling_order=[], # Empty initially
            operands_to_prefetch=self.operands_to_prefetch
        )
        # Pre-run init steps if any (like prefetch)
        # init_scheduler.prefetch_constant_operands()

        init_available = [n for n, d in self.workload.in_degree() if d == 0]
        init_available.sort(key=lambda x: (x.id, x.sub_id))
        
        # Beam tuple: (order_list, scheduled_set, available_node_list, (latency, energy), scheduler_state)
        beam = [([], set(), init_available, (0.0, 0.0), init_scheduler)]
        
        total_nodes = self.workload.number_of_nodes()
        
        for step in range(total_nodes):
            candidates_for_next_beam = []
            
            if time.time() - start_time > max_time:
                logger.warning("Max search time reached. Returning topological order.")
                return [(n.id, n.sub_id) for n in nx.lexicographical_topological_sort(self.workload)]

            # Prepare all potential candidates
            # We group by parent beam item to batch process if possible
            tasks_input_data = [] # List of (new_order, new_scheduled, new_available, scheduler_state) stored or reconstructed

            if parallel:
                # Parallel Execution with Batching
                # Group tasks by beam item (same scheduler state)
                batches_to_submit = []
                
                # We need to reconstruct the candidate info (scheduled, available) after evaluation
                # So we store metadata: (new_order, new_scheduled, new_available)
                metadata_per_order = {} 

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for order, scheduled, available, current_cost, scheduler_state in beam:
                        if not available:
                            continue
                        
                        orders_batch = []
                        for node in available:
                            node_id = (node.id, node.sub_id)
                            new_order = order + [node_id]
                            
                            new_scheduled = scheduled.copy()
                            new_scheduled.add(node_id)
                            new_available = [n for n in available if (n.id, n.sub_id) != node_id]
                            for succ in self.workload.successors(node):
                                preds = self.workload.predecessors(succ)
                                if all((p.id, p.sub_id) in new_scheduled for p in preds):
                                    new_available.append(succ)
                            new_available.sort(key=lambda x: (x.id, x.sub_id))
                            
                            # Store metadata to reconstruct candidate later
                            metadata_per_order[tuple(new_order)] = (new_scheduled, new_available)
                            orders_batch.append(new_order)
                        
                        # Submit batch
                        if orders_batch:
                            futures.append(executor.submit(evaluate_batch_job, scheduler_state, orders_batch))
                    
                    # Collect results
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            for latency, energy, new_scheduler in batch_results:
                                new_order = new_scheduler.scheduling_order
                                new_scheduled, new_available = metadata_per_order[tuple(new_order)]
                                candidates_for_next_beam.append((new_order, new_scheduled, new_available, (latency, energy), new_scheduler))
                        except Exception as e:
                            logger.error(f"Error in parallel evaluation: {e}")

            else:
                # Sequential Execution
                t_seq_start = time.time()
                copy_times = []
                schedule_times = []
                
                for order, scheduled, available, current_cost, scheduler_state in beam:
                    if not available:
                        continue
                    
                    for node in available:
                        node_id = (node.id, node.sub_id)
                        
                        new_order = order + [node_id]
                        new_scheduled = scheduled.copy()
                        new_scheduled.add(node_id)
                        
                        new_available = [n for n in available if (n.id, n.sub_id) != node_id]
                        for succ in self.workload.successors(node):
                            preds = self.workload.predecessors(succ)
                            if all((p.id, p.sub_id) in new_scheduled for p in preds):
                                new_available.append(succ)
                        new_available.sort(key=lambda x: (x.id, x.sub_id))
                        
                        # Timestamp 1
                        ts1 = time.time()
                        # Incremental evaluation (uses evaluate_order_job under the hood)
                        latency, energy, new_scheduler = self.evaluate_partial_order_incremental(new_order, scheduler_state)
                        ts2 = time.time()
                        
                        candidates_for_next_beam.append((new_order, new_scheduled, new_available, (latency, energy), new_scheduler))
                
                # if step % 10 == 0:
                #    logger.info(f"Step {step} processing time: {time.time()-t_seq_start:.4f}s")

            if not candidates_for_next_beam:
                break
                
            candidates_for_next_beam.sort(key=lambda x: (x[3][0], x[3][1]))
            beam = candidates_for_next_beam[:beam_width] # Pick top K
            
            best_latency = beam[0][3][0]
            best_energy = beam[0][3][1]
            logger.info(f"Step {step + 1}/{total_nodes} complete. Best Latency: {best_latency:.0f}, Best Energy: {best_energy:.1f}")
            
        logger.info(f"Baseline Topological Latency: {base_latency:.0f}, Energy: {base_energy:.1f}")
        return beam[0][0]
        