# 11: Execution

## Hardware Requirements and Routing
ALM requires significant computational horsepower. 

### Local Execution (MacBook / Apple Silicon)
- **Use Case:** Code development, schema testing, and single-file mock evaluation.
- **Backend:** MPS (Metal Performance Shaders).
- **Limitation:** MPS does not support `int8_float16` quantization required by `faster-whisper`, meaning execution is extremely slow and memory-intensive. `compute_precision` must fall back to `"auto"` or `"float32"`.

### Production Execution (Google Colab / Cloud GPU)
- **Use Case:** Full 250-sample HOASU-Bench evaluation and scientific CSV generation.
- **Backend:** CUDA (L4 or A100 GPU).
- **Advantage:** Native support for `float16` and flash-attention, reducing inference times from minutes to seconds.
- **Required Commands:** Executing the `colab_setup.ipynb` notebook handles environment setup, pip installs, and GitHub cloning automatically.

## Expected Outputs and Logging
Running `python main.py samples/test_audio.wav` will stream logging directly to the console. The user will see:
1. `[INFO] Neural Perception... Complete.`
2. `[INFO] Validating AudioEvidenceObject... Passed.`
3. `[INFO] Executing WSE...`
Finally, the HOASU Markdown report is printed to standard out and saved to the disk.

## Common Errors
- **CUDA OOM (Out of Memory):** Occurs if Whisper and Qwen are loaded simultaneously without proper offloading. **Solution:** Ensure `del model` and `torch.cuda.empty_cache()` are called sequentially in the pipeline.
