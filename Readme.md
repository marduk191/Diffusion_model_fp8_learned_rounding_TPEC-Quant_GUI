<img width="797" height="728" alt="image" src="https://github.com/user-attachments/assets/8a884d49-d18c-451e-a804-8b99a1decbc5" />

# FP8 Model Quantization Tool

A powerful tool for converting PyTorch models (safetensors format) to FP8 quantized format using learned rounding techniques inspired by AdaRound. This tool provides both command-line and GUI interfaces for easy model quantization.

## Features

- **Advanced Quantization**: Uses SVD-based spectral norm scaling and learned rounding for optimal quantization
- **FP8 Support**: Converts models to `torch.float8_e4m3fn` format for improved inference efficiency
- **Bias Correction**: Automatic bias adjustment to compensate for quantization errors
- **Memory Efficient**: Optimized loading and saving to handle large models
- **Model Compatibility**: Special modes for T5XXL and distillation models
- **Dual Interface**: Both command-line script and user-friendly GUI
- **Stochastic Rounding**: Manual implementation for better quantization quality

## Installation


### Install Dependencies

```
pip install -r requirements.txt
```

### Verify FP8 Support

```python
import torch
try:
    torch.zeros(1, dtype=torch.float8_e4m3fn)
    print("FP8 support available!")
except:
    print("FP8 support not available. Please update PyTorch.")
```

## Usage

### GUI Interface (Recommended)

Run the graphical interface for easy model quantization:

```
python fp8_quantization_gui.py
```

**GUI Features:**
- File browser for input/output selection
- Real-time progress monitoring
- Parameter adjustment with spinboxes
- Settings save/load functionality
- Live output logging
- Automatic output filename generation

### Command Line Interface

For batch processing or scripting:

```
python convert_fp8_scaled_learned_stochastic_bias_svdv2_memeff.py \
    --input model.safetensors \
    --output model_fp8.safetensors \
    --calib_samples 1024 \
    --num_iter 256 \
    --lr 0.02 \
    --reg_lambda 1.0
```

#### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | Required | Input safetensors file path |
| `--output` | Auto-generated | Output safetensors file path |
| `--calib_samples` | 1024 | Number of calibration samples for bias correction |
| `--num_iter` | 256 | Optimization iterations per tensor |
| `--lr` | 0.02 | Learning rate for rounding optimizer |
| `--reg_lambda` | 1.0 | Regularization strength |
| `--t5xxl` | False | Enable T5XXL compatibility mode |
| `--keep_distillation` | False | Preserve distillation layers unquantized |

## Technical Details

### Quantization Method

This tool implements a sophisticated quantization approach:

1. **SVD-based Scaling**: Uses spectral norm (largest singular value) for optimal scale factor calculation
2. **Learned Rounding**: Optimizes rounding decisions using gradient-based methods
3. **Bias Correction**: Automatically adjusts biases to compensate for quantization errors
4. **Stochastic Rounding**: Custom implementation for improved quantization quality

### Memory Efficiency

- **Streaming Processing**: Processes tensors one at a time to minimize memory usage
- **Efficient I/O**: Custom safetensors loading/saving optimized for large files
- **GPU Memory Management**: Automatic cleanup and cache clearing

### Model Compatibility

- **T5XXL Mode**: Excludes decoder layers and certain parameter types
- **Distillation Support**: Option to preserve distillation layers in original precision
- **General Models**: Works with standard PyTorch models in safetensors format

## File Structure

```
fp8-quantization-tool/
├── convert_fp8_scaled_learned_stochastic_bias_svdv2_memeff.py  # Core conversion script
├── fp8_quantization_gui.py                                     # GUI interface
├── requirements.txt                                            # Python dependencies
├── README.md                                                   # This file
└── fp8_gui_settings.json                                      # GUI settings (auto-generated)
```

## Performance Notes

### Hardware Requirements

- **GPU**: CUDA-capable GPU recommended for performance
- **RAM**: Minimum 16GB, 32GB+ recommended for large models
- **Storage**: SSD recommended for faster I/O operations

### Optimization Tips

- Use fewer calibration samples (`--calib_samples`) for faster processing
- Reduce iterations (`--num_iter`) if quality is acceptable
- Enable T5XXL mode for compatible models to reduce processing time
- Monitor GPU memory usage during processing

## Model Quality

The quantized models maintain high quality through:

- **Adaptive Scaling**: Per-tensor scale factors optimized for each layer
- **Learned Rounding**: Minimizes quantization error through optimization
- **Bias Compensation**: Corrects for systematic quantization bias
- **Spectral Normalization**: Preserves important spectral properties

## Troubleshooting

### Common Issues

1. **FP8 Not Supported**
   - Solution: Update to PyTorch build with FP8 support

2. **Out of Memory**
   - Solution: Reduce `calib_samples` or use CPU processing
   - Close other applications using GPU memory

3. **Import Errors**
   - Solution: Ensure all files are in the same directory
   - Check that all dependencies are installed

4. **Slow Processing**
   - Solution: Use GPU acceleration
   - Reduce `num_iter` for faster processing
   - Use SSD storage for model files

### Error Messages

- **"Tensor is all zeros, skipping optimization"**: Normal for empty tensors
- **"No calibration data found"**: Internal error, check model structure
- **"Unsupported float8 type"**: Update PyTorch version

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request


## Acknowledgments

- Inspired by the AdaRound paper: "Up or Down? Adaptive Rounding for Post-Training Quantization"
- Memory-efficient loading/saving and GUI code by marduk191
- Core quantization implementation by Clybius
- initial merging and raw tensor fix by silveroxides


## Changelog

### Version 1.0.0
- Initial release with learned rounding quantization
- GUI interface implementation
- SVD-based scaling method
- Bias correction functionality
- T5XXL and distillation model support
