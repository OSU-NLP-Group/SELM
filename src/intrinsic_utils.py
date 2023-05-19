"""
Thin wrapper around intrinsic module to provide a intrinsic_dimension_vector property on the module.
"""
import torch

import intrinsic

from . import accelerate, config, modeling_utils, util


class IntrinsicDimension(
    intrinsic.IntrinsicDimension,
    modeling_utils.IntrinsicDimension,
    modeling_utils.KnowsBatchSize,
    modeling_utils.Saveable,
):
    @property
    def get_intrinsic_dimension_vector(self):
        return self.intrinsic_vector.detach().cpu()

    def set_intrinsic_dimension_vector(self, vec: torch.Tensor) -> None:
        self.intrinsic_vector.copy_(vec)

    def save(self, path) -> None:
        data = {
            "fastfood_seed": self.seed,
            "theta_d": self.get_intrinsic_dimension_vector.detach(),
        }

        torch.save(data, path)

    def batch_size(self, training_config: config.TrainingConfig) -> int:
        accelerate._set_environment(self)
        self.logger.info("TODO: Take model.context_window into account.")
        if torch.cuda.device_count() < 1:
            self.logger.warn("On CPU; use as big a batch size as you want!")
            return training_config.batch_size

        mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

        self.logger.info(
            "[available memory: %s, rtx2080ti estimate: %s, v100 estimate: %s]",
            mb,
            util.rtx2080ti,
            util.v100,
        )

        model_size = "gpt2"
        try:
            layer_count = len(self.hidden.transformer.h)
            if layer_count == 36:
                model_size = "gpt2-large"
            elif layer_count == 24:
                model_size = "gpt2-medium"
            else:
                assert layer_count == 12
        except AttributeError:
            pass

        if model_size == "gpt2":
            if mb <= util.rtx2080ti:
                # max on rtx2080ti is 2
                assert (
                    accelerate._ENVIRONMENT is accelerate.TrainingType.MODEL_PARALLELISM
                )
                return min(2, training_config.batch_size)
            elif util.rtx2080ti < mb <= util.v100:
                assert accelerate._ENVIRONMENT is accelerate.TrainingType.SINGLE_GPU
                return min(2, training_config.batch_size)
            elif (
                util.v100 < mb <= util.v100 * 2
            ):  # some of the pitzer clusters have 2 NVLINKed v100s.
                assert accelerate._ENVIRONMENT is accelerate.TrainingType.SINGLE_GPU
                return min(4, training_config.batch_size)
            else:
                assert mb > 2 * util.v100
                # deal with this when the time comes
                return training_config.batch_size
        elif model_size == "gpt2-medium":
            if mb < util.a6000:
                assert accelerate._ENVIRONMENT is accelerate.TrainingType.SINGLE_GPU
                return min(2, training_config.batch_size)
        elif model_size == "gpt2-large":
            if mb < util.a6000:
                assert accelerate._ENVIRONMENT is accelerate.TrainingType.SINGLE_GPU
                return 1

        raise ValueError(mb, model_size)
