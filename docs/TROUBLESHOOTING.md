# Troubleshooting

- OOM: reduce batch size and max sequence length.
- Slow training: increase dataloader workers, verify GPU utilization.
- Validation failures: run dataset validator and inspect schema errors.
