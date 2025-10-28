from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from datasets.dagger import DaggerTrajectoryDataset
from algorithms.foundation_dagger import (
    FoundationDaggerModule,
    FoundationDaggerCollector,
)
from experiments.base_exp import BaseLightningExperiment
from experiments.data_modules import BaseDataModule


class FoundationDaggerExperiment(BaseLightningExperiment):
    """
    Experiment that mixes BC training with DFoT-backed DAgger collection.
    """

    compatible_algorithms = dict(
        foundation_dagger=FoundationDaggerModule,
    )
    compatible_datasets = dict(
        dagger=DaggerTrajectoryDataset,
    )
    data_module_cls = BaseDataModule

    def __init__(
        self,
        root_cfg: DictConfig,
        logger=None,
        ckpt_path=None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)

    def collect(self) -> None:
        collector_cfg = self.cfg.collector
        collector = FoundationDaggerCollector(
            policy_cfg=self.root_cfg.algorithm,
            dataset_cfg=collector_cfg.dataset,
            world_model_cfg=collector_cfg.world_model,
            planner_cfg=collector_cfg.planner,
            vlm_cfg=collector_cfg.vlm,
            collector_cfg=collector_cfg.runner,
        )
        output_path = collector.run()
        from omegaconf import open_dict

        with open_dict(self.cfg.collector):
            self.cfg.collector.last_snapshot = str(output_path)
