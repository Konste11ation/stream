import logging
from typing import Any
from stream.stages.stage import Stage, StageCallable
from stream.parser.dvfs_parser import DvfsParser
from stream.workload.onnx_workload import ComputationNodeWorkload
logger = logging.getLogger(__name__)

class DvfsParserStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        dvfs_path: str,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.dvfs_path = dvfs_path
        self.dvfs_parser = DvfsParser(dvfs_path)
    def run(self):
        dvfs_luts = self.dvfs_parser.run()
        self.set_dvfs_data(dvfs_luts)
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            workload = self.workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
    def set_dvfs_data(self, dvfs_luts):
        for node in self.workload.node_list:
            node.set_vdd_lut(dvfs_luts["vdd_lut"])
            node.set_freq_lut(dvfs_luts["freq_lut"])
            node.set_energy_lut(dvfs_luts["energy_lut"])