import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns
from pathlib import Path

class AttentionVisualizer:
    """Visualize attention maps from DMBNet2dv4 model."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.attention_maps = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention outputs."""
        
        def hook_cbam(name):
            def hook(module, input, output):
                self.attention_maps[f'{name}_cbam_input'] = input[0].detach().cpu()
                self.attention_maps[f'{name}_cbam'] = output.detach().cpu()
            return hook
        
        def hook_fourier(name):
            def hook(module, input, output):
                self.attention_maps[f'{name}_fourier_input'] = input[0].detach().cpu()
                self.attention_maps[f'{name}_fourier'] = output.detach().cpu()
            return hook
        
        def hook_fourier_fft(name):
            def hook(module, input, output):
                # Capture input for frequency analysis
                x = input[0]
                # Store original input in frequency domain
                x_fft = torch.fft.rfft2(x, dim=(-2, -1))
                fft_magnitude_input = torch.abs(x_fft).detach().cpu()
                
                # Store output in frequency domain
                output_fft = torch.fft.rfft2(output, dim=(-2, -1))
                fft_magnitude_output = torch.abs(output_fft).detach().cpu()
                
                self.attention_maps[f'{name}_fft_input'] = fft_magnitude_input
                self.attention_maps[f'{name}_fft_output'] = fft_magnitude_output
            return hook
        
        def hook_hybrid(name):
            def hook(module, input, output):
                # Capture input and output for ratio calculation
                self.attention_maps[f'{name}_hybrid_input'] = input[0].detach().cpu()
                # Also capture the gate value
                gate_val = torch.sigmoid(module.gate).detach().cpu()
                self.attention_maps[f'{name}_gate'] = gate_val
                self.attention_maps[f'{name}_hybrid'] = output.detach().cpu()
            return hook
        
        # Register hooks for each block's attention modules
        for i, block in enumerate(self.model.backbone_blocks):
            if hasattr(block, 'attention'):
                # Hook the hybrid attention output
                hook = self.hooks.append(
                    block.attention.register_forward_hook(hook_hybrid(f'block_{i}'))
                )
                
                # Hook CBAM and Fourier attention separately
                self.hooks.append(
                    block.attention.cbam.register_forward_hook(hook_cbam(f'block_{i}'))
                )
                self.hooks.append(
                    block.attention.fourier_attention.register_forward_hook(
                        hook_fourier(f'block_{i}')
                    )
                )
                # Hook Fourier attention for frequency analysis
                self.hooks.append(
                    block.attention.fourier_attention.register_forward_hook(
                        hook_fourier_fft(f'block_{i}')
                    )
                )
    
    def __del__(self):
        """Remove hooks when visualizer is destroyed."""
        for hook in self.hooks:
            hook.remove()
    
    def visualize_attention(
        self, 
        input_tensor: torch.Tensor, 
        save_dir: str = "attention_plots",
        figsize: Tuple[int, int] = (5, 4),
        max_num_cols: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Generate attention visualizations for the given input.
        
        Args:
            input_tensor: Input tensor of shape (B, C, H, W)
            save_dir: Directory to save plots
            figsize: Figure size per subplot
            max_num_cols: Maximum number of columns in the grid
            
        Returns:
            Dictionary containing all attention maps
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Forward pass to capture attention maps
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        
        # Visualize each attention type
        self._plot_attention_maps(save_path, figsize, max_num_cols)
        self._plot_gate_values(save_path, figsize)
        self._plot_attention_comparison(input_tensor.cpu(), save_path, figsize)
        self._plot_attention_ratios(input_tensor.cpu(), save_path, figsize, max_num_cols)
        self._plot_fourier_frequency_analysis(input_tensor.cpu(), save_path, figsize, max_num_cols)
        
        return self.attention_maps
    
    def _plot_attention_maps(self, save_path: Path, figsize: Tuple[int, int], max_num_cols: int):
        """Plot attention maps for each block and attention type."""
        
        attention_types = ['cbam', 'fourier', 'hybrid']
        
        for att_type in attention_types:
            # Get all attention maps of this type
            maps = {k: v for k, v in self.attention_maps.items() if att_type in k and 'gate' not in k}
            
            if not maps:
                continue
                
            n_blocks = len(maps)
            # Calculate grid dimensions
            n_cols = min(n_blocks, max_num_cols)
            n_rows = (n_blocks + n_cols - 1) // n_cols  # Ceiling division
            
            # Calculate total figure size based on subplot size
            total_figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
            
            # Handle case where we have only one subplot
            if n_blocks == 1:
                axes = [axes]
            elif n_rows == 1:
                # If only one row, axes is 1D
                pass
            else:
                # Flatten axes array for easy indexing
                axes = axes.flatten()
            
            fig.suptitle(f'{att_type.upper()} Attention Maps', fontsize=16)
            
            for i, (name, att_map) in enumerate(maps.items()):
                # Average across channels and batch for visualization
                att_avg = att_map.mean(dim=(0, 1)).numpy()
                
                im = axes[i].imshow(att_avg, cmap='Blues', aspect='auto')
                axes[i].set_title(name)
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_blocks, n_rows * n_cols):
                if i < len(axes):
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path / f'{att_type}_attention_maps.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_gate_values(self, save_path: Path, figsize: Tuple[int, int]):
        """Plot gate values showing CBAM vs Fourier attention balance."""
        
        gate_values = {k: v for k, v in self.attention_maps.items() if 'gate' in k}
        
        if not gate_values:
            return
        
        blocks = list(gate_values.keys())
        values = [gate_values[block].item() for block in blocks]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(blocks)), values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Block')
        ax.set_ylabel('Gate Value (0=Fourier, 1=CBAM)')
        ax.set_title('Attention Gate Values: CBAM vs Fourier Balance')
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'gate_values.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_comparison(self, input_tensor: torch.Tensor, save_path: Path, figsize: Tuple[int, int]):
        """Compare input with attention-weighted outputs."""
        
        # Get the first sample from batch for visualization
        input_sample = input_tensor[0]
        
        # Plot original input (average across channels)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Original input
        if input_sample.shape[0] > 1:
            input_avg = input_sample.mean(dim=0).numpy()
        else:
            input_avg = input_sample[0].numpy()
        
        im1 = axes[0, 0].imshow(input_avg, cmap='Blues', aspect='auto')
        axes[0, 0].set_title('Original Input')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Plot attention outputs for the last block (most refined features)
        attention_types = ['cbam', 'fourier', 'hybrid']
        last_block_idx = max([int(k.split('_')[1]) for k in self.attention_maps.keys() 
                             if 'block_' in k and 'gate' not in k])
        
        for i, att_type in enumerate(attention_types):
            key = f'block_{last_block_idx}_{att_type}'
            if key in self.attention_maps:
                att_map = self.attention_maps[key][0].mean(dim=0).numpy()
                
                row, col = (0, i+1) if i < 2 else (1, i-2)
                im = axes[row, col].imshow(att_map, cmap='Blues', aspect='auto')
                axes[row, col].set_title(f'{att_type.upper()} Output')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Plot difference maps
        if 'hybrid' in [k.split('_')[-1] for k in self.attention_maps.keys()]:
            cbam_key = f'block_{last_block_idx}_cbam'
            fourier_key = f'block_{last_block_idx}_fourier'
            
            if cbam_key in self.attention_maps and fourier_key in self.attention_maps:
                cbam_map = self.attention_maps[cbam_key][0].mean(dim=0).numpy()
                fourier_map = self.attention_maps[fourier_key][0].mean(dim=0).numpy()
                diff_map = cbam_map - fourier_map
                
                im = axes[1, 1].imshow(diff_map, cmap='RdBu_r', aspect='auto')
                axes[1, 1].set_title('CBAM - Fourier Difference')
                axes[1, 1].axis('off')
                plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'attention_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_ratios(self, input_tensor: torch.Tensor, save_path: Path, figsize: Tuple[int, int], max_num_cols: int):
        """Plot attention maps as ratios between attention output and input."""
        
        attention_types = ['cbam', 'fourier', 'hybrid']
        
        for att_type in attention_types:
            # Get all attention maps and their corresponding inputs of this type
            output_maps = {k: v for k, v in self.attention_maps.items() 
                          if att_type in k and 'input' not in k and 'gate' not in k and 'fft' not in k}
            input_maps = {k.replace(f'_{att_type}', f'_{att_type}_input'): v 
                         for k, v in output_maps.items() 
                         if k.replace(f'_{att_type}', f'_{att_type}_input') in self.attention_maps}
            
            if not output_maps or not input_maps:
                continue
                
            n_blocks = len(output_maps)
            # Calculate grid dimensions
            n_cols = min(n_blocks, max_num_cols)
            n_rows = (n_blocks + n_cols - 1) // n_cols  # Ceiling division
            
            # Calculate total figure size based on subplot size
            total_figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
            
            # Handle case where we have only one subplot
            if n_blocks == 1:
                axes = [axes]
            elif n_rows == 1:
                # If only one row, axes is 1D
                pass
            else:
                # Flatten axes array for easy indexing
                axes = axes.flatten()
            
            fig.suptitle(f'{att_type.upper()} Attention Ratios (Output/Input)', fontsize=16)
            
            for i, (output_name, output_map) in enumerate(output_maps.items()):
                input_name = output_name.replace(f'_{att_type}', f'_{att_type}_input')
                
                if input_name not in self.attention_maps:
                    continue
                
                input_map = self.attention_maps[input_name]
                
                # Average across channels and batch for visualization
                input_avg = input_map[0].mean(dim=0)  # First sample, average across channels
                output_avg = output_map[0].mean(dim=0)  # First sample, average across channels
                
                # Add small epsilon to avoid division by zero
                epsilon = 1e-8
                input_avg_safe = input_avg + epsilon
                
                # Calculate ratio: attention_output / attention_input
                ratio_map = output_avg / input_avg_safe
                ratio_numpy = ratio_map.numpy()
                
                # Use a diverging colormap centered at 1.0 (no change)
                vmin = max(0.1, ratio_numpy.min())  # Avoid extreme negative values
                vmax = min(5.0, ratio_numpy.max())   # Cap extreme positive values
                
                im = axes[i].imshow(ratio_numpy, cmap='RdBu_r', aspect='auto', 
                                  vmin=vmin, vmax=vmax)
                axes[i].set_title(f'{output_name}\n(Ratio: {ratio_numpy.mean():.2f}±{ratio_numpy.std():.2f})')
                axes[i].axis('off')
                
                # Create colorbar with custom ticks
                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.set_label('Ratio (Output/Input)', rotation=270, labelpad=15)
                
                # Add horizontal line at ratio=1 if within range
                if vmin <= 1.0 <= vmax:
                    # Add text annotation for ratio=1
                    cbar.ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Hide unused subplots
            for i in range(n_blocks, n_rows * n_cols):
                if i < len(axes):
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path / f'{att_type}_attention_ratios.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Also create a summary plot comparing all attention types for the last block
        self._plot_ratio_summary(save_path, figsize)
    
    def _plot_ratio_summary(self, save_path: Path, figsize: Tuple[int, int]):
        """Create a summary plot comparing attention ratios across all types for the last block using correct input/output pairs."""
        
        # Find the last block
        last_block_idx = max([int(k.split('_')[1]) for k in self.attention_maps.keys() 
                             if 'block_' in k and 'gate' not in k])
        
        attention_types = ['cbam', 'fourier', 'hybrid']
        valid_types = []
        ratio_maps = []
        input_maps = []
        
        for att_type in attention_types:
            output_key = f'block_{last_block_idx}_{att_type}'
            input_key = f'block_{last_block_idx}_{att_type}_input'
            
            if output_key in self.attention_maps and input_key in self.attention_maps:
                # Get attention module input and output
                att_input = self.attention_maps[input_key][0].mean(dim=0)  # First sample, average across channels
                att_output = self.attention_maps[output_key][0].mean(dim=0)  # First sample, average across channels
                
                # Calculate ratio between attention output and attention input
                epsilon = 1e-8
                att_input_safe = att_input + epsilon
                ratio_map = att_output / att_input_safe
                
                ratio_maps.append(ratio_map.numpy())
                input_maps.append(att_input.numpy())
                valid_types.append(att_type)
        
        if not ratio_maps:
            return
        
        n_types = len(valid_types)
        fig, axes = plt.subplots(1, n_types + 1, figsize=(figsize[0] * (n_types + 1), figsize[1]))
        
        # Plot the attention input (should be similar across attention types within the same block)
        input_numpy = input_maps[0]  # Use first attention input as reference
        im0 = axes[0].imshow(input_numpy, cmap='Blues', aspect='auto')
        axes[0].set_title('Attention Input')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Find global vmin and vmax for consistent scaling
        all_ratios = np.concatenate([rm.flatten() for rm in ratio_maps])
        vmin = max(0.1, np.percentile(all_ratios, 5))   # 5th percentile, minimum 0.1
        vmax = min(5.0, np.percentile(all_ratios, 95))  # 95th percentile, maximum 5.0
        
        # Plot each attention type ratio
        for i, (att_type, ratio_map) in enumerate(zip(valid_types, ratio_maps)):
            im = axes[i + 1].imshow(ratio_map, cmap='RdBu_r', aspect='auto', 
                                  vmin=vmin, vmax=vmax)
            axes[i + 1].set_title(f'{att_type.upper()} Ratio\n(μ={ratio_map.mean():.2f}, σ={ratio_map.std():.2f})')
            axes[i + 1].axis('off')
            
            cbar = plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
            cbar.set_label('Output/Input Ratio', rotation=270, labelpad=15)
            
            # Add horizontal line at ratio=1 if within range
            if vmin <= 1.0 <= vmax:
                cbar.ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.suptitle(f'Attention Module Ratio Summary - Block {last_block_idx}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path / 'attention_ratio_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_fourier_frequency_analysis(self, input_tensor: torch.Tensor, save_path: Path, figsize: Tuple[int, int], max_num_cols: int):
        """Plot frequency domain analysis showing how Fourier attention affects different frequencies."""
        
        # Get FFT data for all blocks
        fft_input_maps = {k: v for k, v in self.attention_maps.items() if 'fft_input' in k}
        fft_output_maps = {k: v for k, v in self.attention_maps.items() if 'fft_output' in k}
        
        if not fft_input_maps or not fft_output_maps:
            return
        
        # Plot frequency magnitude changes for each block
        n_blocks = len(fft_input_maps)
        n_cols = min(n_blocks, max_num_cols)
        n_rows = (n_blocks + n_cols - 1) // n_cols
        
        total_figsize = (figsize[0] * n_cols, figsize[1] * n_rows * 2)
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=total_figsize)
        
        if n_blocks == 1:
            axes = axes.reshape(-1, 1)
        elif n_rows == 1:
            axes = axes.reshape(2, -1)
        
        fig.suptitle('Fourier Attention: Frequency Domain Analysis', fontsize=16)
        
        for i, block_name in enumerate(sorted(fft_input_maps.keys())):
            col = i % n_cols
            
            # Get corresponding output
            output_name = block_name.replace('fft_input', 'fft_output')
            if output_name not in fft_output_maps:
                continue
            
            # Average across batch and channels for visualization
            fft_input = fft_input_maps[block_name][0].mean(dim=0).numpy()
            fft_output = fft_output_maps[output_name][0].mean(dim=0).numpy()
            
            # Calculate frequency ratio (output/input)
            epsilon = 1e-8
            freq_ratio = fft_output / (fft_input + epsilon)
            
            # Plot input frequency magnitude
            row_input = (i // n_cols) * 2
            im1 = axes[row_input, col].imshow(fft_input, cmap='viridis', aspect='auto')
            axes[row_input, col].set_title(f'{block_name.replace("_fft_input", "")} - Input FFT Magnitude')
            axes[row_input, col].axis('off')
            plt.colorbar(im1, ax=axes[row_input, col], fraction=0.046, pad=0.04)
            
            # Plot frequency change ratio
            row_ratio = row_input + 1
            im2 = axes[row_ratio, col].imshow(freq_ratio, cmap='RdBu_r', aspect='auto',
                                            vmin=0.5, vmax=2.0)  # Center around 1.0
            axes[row_ratio, col].set_title(f'Frequency Ratio (Output/Input)\n(μ={freq_ratio.mean():.2f})')
            axes[row_ratio, col].axis('off')
            cbar = plt.colorbar(im2, ax=axes[row_ratio, col], fraction=0.046, pad=0.04)
            cbar.set_label('Frequency Ratio', rotation=270, labelpad=15)
            
            # Add reference line at ratio=1
            cbar.ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # Hide unused subplots
        total_subplots = n_rows * 2 * n_cols
        for i in range(n_blocks * 2, total_subplots):
            row = i // n_cols
            col = i % n_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'fourier_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create frequency spectrum plots (1D analysis)
        self._plot_frequency_spectrum_analysis(save_path, figsize)
    
    def _plot_frequency_spectrum_analysis(self, save_path: Path, figsize: Tuple[int, int]):
        """Create 1D frequency spectrum plots showing how different frequency components are affected."""
        
        fft_input_maps = {k: v for k, v in self.attention_maps.items() if 'fft_input' in k}
        fft_output_maps = {k: v for k, v in self.attention_maps.items() if 'fft_output' in k}
        
        if not fft_input_maps or not fft_output_maps:
            return
        
        # Get the last block for detailed analysis
        last_block_key = max(fft_input_maps.keys(), key=lambda x: int(x.split('_')[1]))
        output_key = last_block_key.replace('fft_input', 'fft_output')
        
        if output_key not in fft_output_maps:
            return
        
        # Average across batch and channels
        fft_input = fft_input_maps[last_block_key][0].mean(dim=0).numpy()
        fft_output = fft_output_maps[output_key][0].mean(dim=0).numpy()
        
        # Calculate radial frequency spectrum (average over angles)
        h, w = fft_input.shape
        center_h, center_w = h // 2, w // 2
        
        # Create frequency coordinate arrays
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Define frequency bins
        max_r = min(center_h, center_w)
        r_bins = np.linspace(0, max_r, 50)
        
        # Calculate radial averages for input and output
        input_spectrum = []
        output_spectrum = []
        
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if np.any(mask):
                input_spectrum.append(np.mean(fft_input[mask]))
                output_spectrum.append(np.mean(fft_output[mask]))
            else:
                input_spectrum.append(0)
                output_spectrum.append(0)
        
        input_spectrum = np.array(input_spectrum)
        output_spectrum = np.array(output_spectrum)
        
        # Calculate frequency bin centers
        freq_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        # Create the spectrum plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0] * 1.5, figsize[1] * 2))
        
        # Plot input vs output spectra
        ax1.plot(freq_centers, input_spectrum, 'b-', label='Input Spectrum', linewidth=2)
        ax1.plot(freq_centers, output_spectrum, 'r-', label='Output Spectrum', linewidth=2)
        ax1.set_xlabel('Frequency (normalized)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'Frequency Spectrum Comparison - {last_block_key.replace("_fft_input", "")}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot frequency ratio
        epsilon = 1e-8
        ratio_spectrum = output_spectrum / (input_spectrum + epsilon)
        ax2.plot(freq_centers, ratio_spectrum, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='No Change')
        ax2.set_xlabel('Frequency (normalized)')
        ax2.set_ylabel('Ratio (Output/Input)')
        ax2.set_title('Frequency Attenuation/Amplification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for key frequency regions
        low_freq_idx = len(freq_centers) // 4
        high_freq_idx = 3 * len(freq_centers) // 4
        
        ax2.annotate(f'Low Freq Ratio: {ratio_spectrum[low_freq_idx]:.2f}', 
                    xy=(freq_centers[low_freq_idx], ratio_spectrum[low_freq_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax2.annotate(f'High Freq Ratio: {ratio_spectrum[high_freq_idx]:.2f}', 
                    xy=(freq_centers[high_freq_idx], ratio_spectrum[high_freq_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(save_path / 'frequency_spectrum_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()