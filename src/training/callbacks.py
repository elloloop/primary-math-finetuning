from transformers import TrainerCallback


class SimpleMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"step={state.global_step} logs={logs}")
