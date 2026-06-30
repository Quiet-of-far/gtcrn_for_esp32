# GTCRN paired-data fine-tuning

Run a smoke test:

```bash
bash finetune/run_finetune.sh --smoke --output-dir /tmp/gtcrn_finetune_smoke
```

Start the full run:

```bash
bash finetune/run_finetune.sh
```

Resume an interrupted run using the same output directory:

```bash
bash finetune/run_finetune.sh \
  --resume experiments/gtcrn_dns3_paired_finetune/<run>/checkpoints/latest.tar \
  --output-dir experiments/gtcrn_dns3_paired_finetune/<run>
```

Export the selected checkpoint:

```bash
conda run --no-capture-output -n vinp310 python -m finetune.export \
  --checkpoint experiments/gtcrn_dns3_paired_finetune/<run>/checkpoints/best_pesq.tar \
  --output-dir experiments/gtcrn_dns3_paired_finetune/<run>/export
```
