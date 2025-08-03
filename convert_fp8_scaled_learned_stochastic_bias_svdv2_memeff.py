#!/usr/bin/env python3
import argparse
import os
import torch
from typing import Dict, Tuple, List, Any
from tqdm import tqdm
import gc
import json
import struct
from pathlib import Path

# Written by Clybius, with memory-efficient loading/saving from marduk191

# --- START: Memory-Efficient Loading/Saving Code ---

class MemoryEfficientSafeOpen:
    # does not support metadata loading
    def __init__(self, filename):
        self.filename = filename
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            return torch.empty(shape, dtype=dtype)
        
        tensor_bytes = bytearray(tensor_bytes)  # make it writable
        byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        tensor = byte_tensor.view(dtype).reshape(shape)
        
        # Handle scalar tensors (0-dimensional tensors)
        if len(shape) == 0:
            return tensor.squeeze()
        
        return tensor

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    print(f"\nUsing memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        # Handle scalar tensors (0-dimensional)
        if v.numel() == 0 or (v.dim() == 0):
            if v.dim() == 0:
                # Scalar tensor - store as single element
                size = v.element_size()
                header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
                offset += size
            else:
                # Empty tensor
                header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        pbar = tqdm(tensors.items(), desc="Saving tensors")
        for k, v in pbar:
            # Skip truly empty tensors (numel == 0 but not scalar)
            if v.numel() == 0 and v.dim() > 0:
                continue
                
            # Handle scalar tensors and regular tensors
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # scalar tensor
                        # Create a temporary view for scalar
                        scalar_bytes = v.detach().view(torch.uint8)
                        scalar_bytes.cpu().numpy().tofile(f)
                    else:
                        tensor_bytes = v.contiguous().view(torch.uint8)
                        tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # scalar tensor
                    # Handle scalar tensors properly
                    scalar_bytes = v.detach().view(torch.uint8)
                    scalar_bytes.numpy().tofile(f)
                else:
                    v.contiguous().view(torch.uint8).numpy().tofile(f)

# --- END: Memory-Efficient Loading/Saving Code ---


# Keys containing these strings will not be quantized if --t5xxl is set
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"] #T5XXL, may need to be changed for other TEs.
# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32 # Don't think more hurts here since we're working tensor by tensor.
# Dtype for storing scale factors
SCALE_DTYPE = torch.float32

def find_best_scale(W_float32, f8_max_val):
    """Searches for a scale that minimizes the initial quantization error."""
    w_max = W_float32.abs().max()
    if w_max < 1e-12:
        return torch.tensor(1.0, device=W_float32.device)

    # Search in a small range around the absmax scale
    best_scale = f8_max_val / w_max
    min_err = float('inf')

    # Iterate over a few candidate scales
    for i in range(80, 101, 2): # Search from 80% to 100% of max range
        scale_candidate = (f8_max_val / w_max) * (i / 100.0)
        W_scaled = W_float32 * scale_candidate
        W_quant = W_scaled.to(TARGET_FP8_DTYPE) # Use your TARGET_FP8_DTYPE
        W_dequant = W_quant.to(W_float32.dtype) / scale_candidate
        
        err = (W_float32 - W_dequant).pow(2).mean()
        
        if err < min_err:
            min_err = err
            best_scale = scale_candidate
    
    return best_scale

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign

def stochastic_rounding(value, dtype=TARGET_FP8_DTYPE, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        #num_slices = max(1, (value.numel() / (1536 * 1536)))
        #slice_size = max(1, round(value.shape[0] / num_slices))
        #for i in range(0, value.shape[0], slice_size):
        #    output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
        output.copy_(manual_stochastic_round_to_float8(value, dtype, generator=generator))
        return output

def quantize_weight_svd(W: torch.Tensor, fp8_max: float):
    """Quantizes using the SVD-based spectral norm method."""
    # We only need the singular values, so we can use a more efficient computation
    S = torch.linalg.svdvals(W)
    s_max = S[0] # The largest singular value is the spectral norm

    scale = fp8_max / s_max
    return scale

class LearnedRoundingConverter:
    """
    Implements adaptive rounding for converting a weight to float8.
    Inspired by AdaRound paper (https://arxiv.org/abs/2004.10568).
    """
    def __init__(self, num_iter=500, lr=1e-3, reg_lambda=0.01, beta_start=20, beta_end=2):
        self.num_iter = num_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # The maximum representable value for e4m3fn, used for scaling.
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"LearnedRoundingConverter initialized on device: {self.device}")

    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the learned rounding conversion for a single weight tensor.
        """
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)

        # Step 1: Calculate the quantization scale (per-tensor asymmetric)
        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            print("  - Tensor is all zeros, skipping optimization.")
            scale = torch.tensor(1.0, device=self.device)
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), scale.reciprocal().cpu().reshape(1), torch.zeros_like(W_float32).cpu()

        scale = self.f8_max_val / w_max # Example: (absmax = 1, fp8 max = +-448 for dtype e4m3_fn)
        W_scaled = W_float32 * scale # absmax now +-448

        # Step 2: Initialize the rounding mask 'h'
        W_rounded = W_scaled.to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE) # Naive RtN quantization on scaled model
        W_dq_rounded = W_rounded / scale # Scale back down with scalar
        #error = W_dq_rounded - W_float32 # Calculate error
        U, S, Vh = torch.linalg.svd(W_float32, full_matrices=False) # Do SVD decomposition on original tensor in FP32 format
        U_k = U[:, :1] # Obtain most important low-rank matrices
        Vh_k = Vh[:1, :]

        W_q_refined = W_rounded.clone()

        # Step 4: The optimization loop
        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False)
        for i in pbar:

            current_dq = W_q_refined / scale
            error = current_dq - W_float32

            projected_error = U_k.T @ error @ Vh_k.T

            rounding_direction = torch.sign(W_q_refined - W_scaled)


            candidate_indices = torch.nonzero(rounding_direction, as_tuple=False)

            if len(candidate_indices) == 0:
                break # No more candidates to flip

            G = U_k @ projected_error @ Vh_k
            u_norms_sq = torch.sum(U_k**2, dim=1, keepdim=True)
            vh_norms_sq = torch.sum(Vh_k**2, dim=0, keepdim=True)
            norm_term_matrix = (1.0 / scale**2) * (u_norms_sq @ vh_norms_sq)

            cand_rows = candidate_indices[:, 0]
            cand_cols = candidate_indices[:, 1]

            cand_rd = rounding_direction[cand_rows, cand_cols]
            cand_G = G[cand_rows, cand_cols]
            cand_norm_term = norm_term_matrix[cand_rows, cand_cols]

            objective_values = 2 * (-cand_rd / scale) * cand_G + cand_norm_term

            min_objective_val, min_local_idx = torch.min(objective_values, dim=0)

            if min_objective_val >= 0:
                break
            
            best_flip_global_idx = candidate_indices[min_local_idx]
            row, col = best_flip_global_idx[0], best_flip_global_idx[1]
            
            W_q_refined[row, col] += -rounding_direction[row, col]


        # Final Hard Quantization
        with torch.no_grad():
            W_f8 = W_q_refined.to(torch.float8_e4m3fn)

        # Calculate dequantization scale (reciprocal of the quantization scale)
        dequant_scale = scale.reciprocal().reshape(1)
        # Clean up GPU memory
        del W_float32, W_scaled, W_rounded, W_q_refined, W_dq_rounded, error, U, S, Vh, U_k, Vh_k
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8.to(device=torch.device('cpu')), dequant_scale.to(dtype=SCALE_DTYPE, device=torch.device('cpu')), (W_f8.to(COMPUTE_DTYPE) * dequant_scale).to(device=torch.device('cpu'))

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

# Global FP8 constants
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, calib_samples: int, **converter_kwargs):
    """
    Converts a safetensors file to a version with FP8 scaled weights using learned rounding (modified from AdaRound).
    """
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using FP8 format: {TARGET_FP8_DTYPE}")
    print(f"FP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    print(f"FP8 Min Precision: [{FP8_MIN_POS}]")

    new_tensors: Dict[str, torch.Tensor] = {}

    with MemoryEfficientSafeOpen(input_file) as f:
        all_keys = f.keys()
        all_keys_set = set(all_keys)
        original_tensor_count = len(all_keys)

        # Instantiate the converter with hyperparameters from command line
        converter = LearnedRoundingConverter(**converter_kwargs)

        # Pre-generate calibration data using shapes from the header for efficiency
        print("\nScanning model for linear layer dimensions...")
        calibration_data_cache = {}
        for key in all_keys:
            if key.endswith('.weight'):
                metadata = f.header[key]
                shape = metadata.get('shape', [])
                if len(shape) == 2:
                    in_features = shape[1]
                    if in_features not in calibration_data_cache:
                        print(f"  - Found new in_features dimension: {in_features}. Generating calibration data.")
                        calibration_data_cache[in_features] = torch.randn(
                            calib_samples, in_features, dtype=COMPUTE_DTYPE
                        )
        print("Calibration data generated.\n")

        weight_keys = sorted([key for key in all_keys if key.endswith('.weight')])
        total_weights = len(weight_keys)
        skipped_count = 0
        processed_count = 0

        print(f"Found {total_weights} weight tensors to potentially process.")

        for i, key in enumerate(weight_keys):
            process_this_key = True

            if t5xxl and any(avoid_name in key for avoid_name in ["decoder"]):
                print(f"({i+1}/{total_weights}) Removing decoder T5XXL tensor: {key}")
                process_this_key = False
                skipped_count += 1
                continue

            if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
                print(f"({i+1}/{total_weights}) Skipping excluded T5XXL tensor: {key}")
                new_tensors[key] = f.get_tensor(key)
                process_this_key = False
                skipped_count += 1

            if keep_distillation and any(avoid_name in key for avoid_name in ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]):
                print(f"({i+1}/{total_weights}) Skipping excluded distillation tensor: {key}")
                new_tensors[key] = f.get_tensor(key)
                base_name = key[:-len('.weight')]
                scale_weight_key = f"{base_name}.scale_weight"
                dequant_scale = torch.tensor([1.0], dtype=SCALE_DTYPE)
                new_tensors[scale_weight_key] = dequant_scale.detach().clone()
                process_this_key = False
                skipped_count += 1

            if not process_this_key:
                continue

            print(f"({i+1}/{total_weights}) Processing tensor: {key}")
            processed_count += 1

            original_tensor = f.get_tensor(key)

            if original_tensor.numel() == 0 or original_tensor.ndim != 2:
                print(f"  - Skipping empty or non-2D tensor: {key}")
                new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE) # Store as empty FP8
                base_name = key[:-len('.weight')]
                scale_weight_key = f"{base_name}.scale_weight"
                dequant_scale = torch.tensor([1.0], dtype=SCALE_DTYPE)
                new_tensors[scale_weight_key] = dequant_scale.detach().clone()
                continue

            in_features = original_tensor.shape[1]
            if in_features not in calibration_data_cache:
                 print(f"  - WARNING: No calibration data found for in_features={in_features}. Skipping {key}")
                 new_tensors[key] = original_tensor
                 skipped_count += 1
                 processed_count -= 1
                 continue

            calibration_data = calibration_data_cache[in_features]

            quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert(original_tensor, calibration_data)

            new_tensors[key] = quantized_fp8_tensor
            base_name = key[:-len('.weight')]
            bias_key = f"{base_name}.bias"
            scale_weight_key = f"{base_name}.scale_weight"
            new_tensors[scale_weight_key] = dequant_scale.detach().clone()

            if bias_key in all_keys_set:
                print(f"  - Found and adjusting corresponding bias: {bias_key}")
                with torch.no_grad():
                    original_bias = f.get_tensor(bias_key)
                    
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    W_orig_dev = original_tensor.to(device, dtype=COMPUTE_DTYPE)
                    W_dequant_dev = dequantized_weight_tensor.to(device, dtype=COMPUTE_DTYPE)
                    X_calib_dev = calibration_data.to(device, dtype=COMPUTE_DTYPE)
                    b_orig_dev = original_bias.to(device, dtype=COMPUTE_DTYPE)

                    weight_error = W_orig_dev - W_dequant_dev
                    output_error = X_calib_dev @ weight_error.T
                    bias_correction = output_error.mean(dim=0)
                    b_new = b_orig_dev - bias_correction
                    
                    new_tensors[bias_key] = b_new.cpu().to(original_bias.dtype)
                    
                    print(f"  - Original bias mean: {original_bias.mean().item():.6f}")
                    print(f"  - New bias mean     : {new_tensors[bias_key].mean().item():.6f}")
                    
                    del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new, original_bias
                    if device == 'cuda':
                        torch.cuda.empty_cache()

            if t5xxl:
                scale_input_key = f"{base_name}.scale_input"
                new_tensors[scale_input_key] = dequant_scale.detach().clone().to(SCALE_DTYPE)

            print(f"  - Dequant Scale  : {dequant_scale.item():.9}")
            
            del original_tensor, quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor
            gc.collect()

        # Add remaining unprocessed tensors to the new file
        print("\nAdding remaining tensors...")
        for key in tqdm(all_keys, desc="Copying other tensors"):
            if t5xxl and "decoder" in key:
                if key not in new_tensors:
                    skipped_count +=1
                continue # Already handled in the loop, this ensures no non-weight decoder tensors are added
            if key not in new_tensors:
                new_tensors[key] = f.get_tensor(key)

    new_tensors["scaled_fp8"] = torch.empty((2), dtype=TARGET_FP8_DTYPE) if not t5xxl else torch.empty((0), dtype=TARGET_FP8_DTYPE)

    print("-" * 40)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Use memory-efficient saving
        mem_eff_save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}")
        return

    print("-" * 40)
    print("Summary:")
    print(f"  - Original tensor count : {original_tensor_count}")
    print(f"  - Weights processed     : {processed_count}")
    print(f"  - Weights skipped       : {skipped_count}")
    print(f"  - Final tensor count    : {len(new_tensors)}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors weights to Scaled {TARGET_FP8_DTYPE} format using learned rounding, adapted from AdaRound.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Original arguments
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. If not provided, generated based on input name.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization. \n(Likely not helpful because ComfyUI may use Round-to-Nearest in place of this, which SUXASS.)")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")

    parser.add_argument("--calib_samples", type=int, default=1024, help="Number of random samples for calibration.") # Random calibration samples for bias correction
    parser.add_argument("--num_iter", type=int, default=256, help="Number of optimization iterations per tensor.") # 256 iterations seems good enough, don't think higher is needed based on results with 256.
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate for the rounding optimizer.")
    parser.add_argument("--reg_lambda", type=float, default=1.00, help="Regularization strength for the rounding loss.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Check for FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("Error: This version of PyTorch or this hardware does not support torch.float8_e4m3fn.")
        return

    fp8_type_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
    distill_str = "_nodistill" if args.keep_distillation else ""
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_{fp8_type_str}_scaled_learned{distill_str}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be the same as the input file.")
        return

    # Pass learned rounding hyperparameters to the conversion function
    converter_kwargs = {
        'num_iter': args.num_iter,
        'lr': args.lr,
        'reg_lambda': args.reg_lambda,
    }

    convert_to_fp8_scaled(
        args.input,
        output_file,
        args.t5xxl,
        args.keep_distillation,
        args.calib_samples,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()