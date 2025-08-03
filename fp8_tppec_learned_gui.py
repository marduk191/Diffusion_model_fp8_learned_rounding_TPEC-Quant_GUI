#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import subprocess
from pathlib import Path
import json

# Import the original conversion functions
try:
    from convert_fp8_scaled_learned_stochastic_bias_svdv2_memeff import convert_to_fp8_scaled, TARGET_FP8_DTYPE
    import torch
    CONVERSION_MODULE_AVAILABLE = True
except ImportError as e:
    CONVERSION_MODULE_AVAILABLE = False
    IMPORT_ERROR = str(e)

class RedirectText:
    """Redirect stdout/stderr to the GUI text widget"""
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert('end', string, self.tag)
        self.text_widget.configure(state='disabled')
        self.text_widget.see('end')
        self.text_widget.update()

    def flush(self):
        pass

class FP8QuantizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FP8 TPEC-Quant Quantization Tool")
        self.root.geometry("800x700")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.keep_distillation = tk.BooleanVar(value=False)
        self.t5xxl = tk.BooleanVar(value=False)
        self.calib_samples = tk.IntVar(value=1024)
        self.num_iter = tk.IntVar(value=256)
        self.lr = tk.DoubleVar(value=0.02)
        self.reg_lambda = tk.DoubleVar(value=1.0)
        
        # State variables
        self.is_processing = False
        self.settings_file = "fp8_gui_settings.json"
        
        self.create_widgets()
        self.load_settings()
        
        # Check if conversion module is available
        if not CONVERSION_MODULE_AVAILABLE:
            self.show_import_error()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="FP8 TPEC-Quant Quantization Tool", 
                               font=('TkDefaultFont', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection section
        self.create_file_section(main_frame, 1)
        
        # Model options section
        self.create_options_section(main_frame, 4)
        
        # Advanced parameters section
        self.create_parameters_section(main_frame, 7)
        
        # Control buttons
        self.create_control_section(main_frame, 10)
        
        # Progress section
        self.create_progress_section(main_frame, 11)
        
        # Log output section
        self.create_log_section(main_frame, 12)

    def create_file_section(self, parent, start_row):
        # File selection section
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Input file
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(file_frame, textvariable=self.input_file, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2)
        
        # Output file
        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        ttk.Entry(file_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_output_file).grid(row=1, column=2, pady=(10, 0))
        
        # Auto-generate output checkbox
        self.auto_output = tk.BooleanVar(value=True)
        ttk.Checkbutton(file_frame, text="Auto-generate output filename", 
                       variable=self.auto_output, command=self.on_auto_output_changed).grid(row=2, column=1, sticky=tk.W, pady=(5, 0))

    def create_options_section(self, parent, start_row):
        # Model options section
        options_frame = ttk.LabelFrame(parent, text="Model Options", padding="10")
        options_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="T5XXL compatibility mode", 
                       variable=self.t5xxl).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Checkbutton(options_frame, text="Keep distillation layers", 
                       variable=self.keep_distillation).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Tooltips (simulated with labels)
        ttk.Label(options_frame, text="• T5XXL mode excludes certain layers for compatibility", 
                 foreground="gray", font=('TkDefaultFont', 8)).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(options_frame, text="• Keeps distillation layers unquantized", 
                 foreground="gray", font=('TkDefaultFont', 8)).grid(row=1, column=1, sticky=tk.W, padx=(20, 0), pady=(5, 0))

    def create_parameters_section(self, parent, start_row):
        # Advanced parameters section
        params_frame = ttk.LabelFrame(parent, text="Quantization Parameters", padding="10")
        params_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        
        # Calibration samples
        ttk.Label(params_frame, text="Calibration Samples:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        calib_spin = ttk.Spinbox(params_frame, from_=128, to=4096, increment=128, textvariable=self.calib_samples, width=10)
        calib_spin.grid(row=0, column=1, sticky=tk.W)
        
        # Number of iterations
        ttk.Label(params_frame, text="Optimization Iterations:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        iter_spin = ttk.Spinbox(params_frame, from_=64, to=1024, increment=64, textvariable=self.num_iter, width=10)
        iter_spin.grid(row=0, column=3, sticky=tk.W)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        lr_spin = ttk.Spinbox(params_frame, from_=0.001, to=0.1, increment=0.001, textvariable=self.lr, width=10, format="%.3f")
        lr_spin.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Regularization lambda
        ttk.Label(params_frame, text="Regularization λ:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0))
        lambda_spin = ttk.Spinbox(params_frame, from_=0.01, to=2.0, increment=0.01, textvariable=self.reg_lambda, width=10, format="%.2f")
        lambda_spin.grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Reset to defaults button
        ttk.Button(params_frame, text="Reset to Defaults", command=self.reset_defaults).grid(row=2, column=0, columnspan=4, pady=(10, 0))

    def create_control_section(self, parent, start_row):
        # Control buttons section
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=start_row, column=0, columnspan=3, pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Quantization", command=self.start_conversion)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_conversion, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT)

    def create_progress_section(self, parent, start_row):
        # Progress section
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

    def create_log_section(self, parent, start_row):
        # Log output section
        log_frame = ttk.LabelFrame(parent, text="Output Log", padding="5")
        log_frame.grid(row=start_row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(start_row, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for different message types
        self.log_text.tag_configure("stdout", foreground="black")
        self.log_text.tag_configure("stderr", foreground="red")
        self.log_text.tag_configure("info", foreground="blue")
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input Safetensors File",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            if self.auto_output.get():
                self.generate_output_filename()

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output File Location",
            defaultextension=".safetensors",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            self.auto_output.set(False)

    def on_auto_output_changed(self):
        if self.auto_output.get() and self.input_file.get():
            self.generate_output_filename()

    def generate_output_filename(self):
        if not self.input_file.get():
            return
        
        input_path = Path(self.input_file.get())
        base_name = input_path.stem
        
        # Generate suffix based on options
        fp8_type_str = "float8_e4m3fn"  # TARGET_FP8_DTYPE string representation
        distill_str = "_nodistill" if self.keep_distillation.get() else ""
        t5xxl_str = "_t5xxl" if self.t5xxl.get() else ""
        
        output_filename = f"{base_name}_{fp8_type_str}_scaled_learned{distill_str}{t5xxl_str}.safetensors"
        output_path = input_path.parent / output_filename
        self.output_file.set(str(output_path))

    def reset_defaults(self):
        self.calib_samples.set(1024)
        self.num_iter.set(256)
        self.lr.set(0.02)
        self.reg_lambda.set(1.0)

    def save_settings(self):
        settings = {
            'keep_distillation': self.keep_distillation.get(),
            't5xxl': self.t5xxl.get(),
            'calib_samples': self.calib_samples.get(),
            'num_iter': self.num_iter.get(),
            'lr': self.lr.get(),
            'reg_lambda': self.reg_lambda.get(),
            'auto_output': self.auto_output.get()
        }
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            self.log_message("Settings saved successfully.", "info")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def load_settings(self):
        if not os.path.exists(self.settings_file):
            return
        
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            
            self.keep_distillation.set(settings.get('keep_distillation', False))
            self.t5xxl.set(settings.get('t5xxl', False))
            self.calib_samples.set(settings.get('calib_samples', 1024))
            self.num_iter.set(settings.get('num_iter', 256))
            self.lr.set(settings.get('lr', 0.02))
            self.reg_lambda.set(settings.get('reg_lambda', 1.0))
            self.auto_output.set(settings.get('auto_output', True))
            
            self.log_message("Settings loaded successfully.", "info")
        except Exception as e:
            self.log_message(f"Failed to load settings: {e}", "stderr")

    def log_message(self, message, tag="stdout"):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', f"{message}\n", tag)
        self.log_text.configure(state='disabled')
        self.log_text.see('end')
        self.root.update()

    def clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, 'end')
        self.log_text.configure(state='disabled')

    def validate_inputs(self):
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file.")
            return False
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return False
        
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file.")
            return False
        
        if os.path.abspath(self.input_file.get()) == os.path.abspath(self.output_file.get()):
            messagebox.showerror("Error", "Output file cannot be the same as input file.")
            return False
        
        return True

    def start_conversion(self):
        if not CONVERSION_MODULE_AVAILABLE:
            self.show_import_error()
            return
        
        if not self.validate_inputs():
            return
        
        # Update UI state
        self.is_processing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_var.set("Starting quantization...")
        self.progress_bar.start()
        
        # Clear log
        self.clear_log()
        
        # Redirect stdout to the log widget
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = RedirectText(self.log_text, "stdout")
        sys.stderr = RedirectText(self.log_text, "stderr")
        
        # Start conversion in a separate thread
        self.conversion_thread = threading.Thread(target=self.run_conversion)
        self.conversion_thread.daemon = True
        self.conversion_thread.start()

    def run_conversion(self):
        try:
            # Prepare converter kwargs
            converter_kwargs = {
                'num_iter': self.num_iter.get(),
                'lr': self.lr.get(),
                'reg_lambda': self.reg_lambda.get(),
            }
            
            # Run the conversion
            convert_to_fp8_scaled(
                self.input_file.get(),
                self.output_file.get(),
                self.t5xxl.get(),
                self.keep_distillation.get(),
                self.calib_samples.get(),
                **converter_kwargs
            )
            
            # Success
            self.root.after(0, self.conversion_completed, True, "Quantization completed successfully!")
            
        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            self.root.after(0, self.conversion_completed, False, error_msg)

    def conversion_completed(self, success, message):
        # Restore stdout/stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        # Update UI state
        self.is_processing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_bar.stop()
        
        if success:
            self.progress_var.set("Quantization completed!")
            self.log_message(message, "info")
            messagebox.showinfo("Success", message)
        else:
            self.progress_var.set("Quantization failed!")
            self.log_message(message, "stderr")
            messagebox.showerror("Error", message)

    def stop_conversion(self):
        # Note: This is a simple implementation. For a more robust solution,
        # you'd need to implement proper thread cancellation in the conversion function
        if hasattr(self, 'conversion_thread') and self.conversion_thread.is_alive():
            self.log_message("Stop requested. Please wait for current operation to complete...", "info")
            # In a real implementation, you'd set a flag that the conversion function checks
        
        self.is_processing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_bar.stop()
        self.progress_var.set("Stopped")

    def show_import_error(self):
        error_msg = ("The conversion module could not be imported. Please ensure:\n\n"
                    "1. The 'convert_fp8_scaled_learned_stochastic_bias_svdv2_memeff.py' file is in the same directory\n"
                    "2. PyTorch is installed with FP8 support\n"
                    "3. All required dependencies are available\n\n"
                    f"Import error: {IMPORT_ERROR}")
        
        messagebox.showerror("Import Error", error_msg)
        self.log_message(error_msg, "stderr")

def main():
    root = tk.Tk()
    app = FP8QuantizationGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()