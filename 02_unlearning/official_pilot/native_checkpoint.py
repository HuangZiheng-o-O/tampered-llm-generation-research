"""Keep ZeRO recovery state in native shards; export weights after training exits."""
from pathlib import Path
from types import MethodType

from transformers import TrainerCallback


class SaveFirstStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_save = True
        return control


def configure_native_checkpoint(trainer, first_step=False):
    if not trainer.is_deepspeed_enabled:
        raise ValueError("Native checkpoint mode requires DeepSpeed")
    config = trainer.accelerator.state.deepspeed_plugin.deepspeed_config
    if config["zero_optimization"]["stage"] != 3 or config["zero_optimization"].get("stage3_gather_16bit_weights_on_model_save", False):
        raise ValueError("Native mode requires ZeRO-3 with weight gathering disabled")

    def save_model(owner, output_dir=None, _internal_call=False):
        output = Path(output_dir or owner.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if owner.args.should_save:
            owner.model.config.save_pretrained(output)
            if getattr(owner.model, "generation_config", None) is not None:
                owner.model.generation_config.save_pretrained(output)
            if owner.processing_class is not None:
                owner.processing_class.save_pretrained(output)
        # HF's checkpoint path next calls _save_optimizer_and_scheduler(),
        # which saves the complete native model/optimizer/scheduler state.
        # A final explicit save_model() needs that call here instead.
        if not _internal_call:
            if owner.model_wrapped.save_checkpoint(str(output)) is False:
                raise RuntimeError("DeepSpeed did not save the final native checkpoint")

    trainer.save_model = MethodType(save_model, trainer)
    if first_step:
        trainer.add_callback(SaveFirstStep())
