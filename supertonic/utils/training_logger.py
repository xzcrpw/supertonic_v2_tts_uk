"""
Beautiful Training Logger for Supertonic v2

Features:
  - Rich console output with colors and tables
  - Real-time metrics dashboard
  - GPU monitoring
  - ETA calculation
  - Loss trend visualization
"""

import os
import sys
import time
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, SpinnerColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class MetricsHistory:
    """Stores metrics history for trend analysis."""
    window_size: int = 100
    g_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    d_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    recon_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    adv_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    fm_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Timing
    step_times: deque = field(default_factory=lambda: deque(maxlen=50))


class TrainingLogger:
    """Beautiful training logger with rich console output."""
    
    def __init__(
        self,
        total_iterations: int,
        log_interval: int = 50,
        checkpoint_interval: int = 1000,
        validation_interval: int = 5000,
        disc_start_steps: int = 5000,
        rank: int = 0,
        world_size: int = 1
    ):
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.disc_start_steps = disc_start_steps
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.history = MetricsHistory()
        
        self.console = Console() if RICH_AVAILABLE and self.is_main else None
        
        if self.is_main:
            self._print_banner()
    
    def _print_banner(self):
        """Print beautiful startup banner."""
        if self.console:
            banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   [bold white]🔊 SUPERTONIC v2[/bold white] - Ukrainian TTS Training                                  ║
║                                                                               ║
║   [yellow]Architecture:[/yellow] WaveNeXt Speech Autoencoder                                   ║
║   [yellow]Paper:[/yellow]        arXiv:2503.23108v3                                            ║
║   [yellow]Stage:[/yellow]        Stage 1 - Speech Autoencoder Training                         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
            self.console.print(banner)
        else:
            print("=" * 80)
            print("🔊 SUPERTONIC v2 - Ukrainian TTS Training")
            print("=" * 80)
    
    def print_config(self, config: Dict[str, Any]):
        """Print training configuration in a nice table."""
        if not self.is_main:
            return
        
        if self.console:
            table = Table(title="Training Configuration", box=box.ROUNDED)
            table.add_column("Parameter", style="cyan", width=30)
            table.add_column("Value", style="green", width=25)
            
            table.add_row("Total Iterations", f"{self.total_iterations:,}")
            table.add_row("Batch Size (per GPU)", str(config.get('batch_size', 'N/A')))
            table.add_row("Effective Batch Size", f"{config.get('batch_size', 0) * self.world_size}")
            table.add_row("Learning Rate", f"{config.get('learning_rate', 'N/A')}")
            table.add_row("GPUs", f"{self.world_size}×")
            table.add_row("", "")
            table.add_row("Checkpoint Every", f"{self.checkpoint_interval:,} steps")
            table.add_row("Validation Every", f"{self.validation_interval:,} steps")
            table.add_row("Log Every", f"{self.log_interval} steps")
            table.add_row("Discriminator Starts", f"Step {self.disc_start_steps:,}")
            table.add_row("", "")
            table.add_row("λ_recon", str(config.get('loss_weights', {}).get('reconstruction', 45)))
            table.add_row("λ_adv", str(config.get('loss_weights', {}).get('adversarial', 1)))
            table.add_row("λ_fm", str(config.get('loss_weights', {}).get('feature_matching', 0.1)))
            
            self.console.print(table)
            print()
        else:
            print(f"Training Config:")
            print(f"  Total iterations: {self.total_iterations:,}")
            print(f"  Batch size: {config.get('batch_size', 'N/A')} × {self.world_size} GPUs")
            print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
    
    def log_step(self, iteration: int, losses: Dict[str, float]):
        """Log a training step with beautiful formatting."""
        if not self.is_main:
            return
        
        # Update history
        self.history.g_loss.append(losses.get('g_loss', 0))
        self.history.d_loss.append(losses.get('d_loss', 0))
        self.history.recon_loss.append(losses.get('recon_loss', 0))
        self.history.adv_loss.append(losses.get('adv_loss', 0))
        self.history.fm_loss.append(losses.get('fm_loss', 0))
        
        # Timing
        current_time = time.time()
        step_time = current_time - self.last_log_time
        self.history.step_times.append(step_time / self.log_interval)
        self.last_log_time = current_time
        
        # Only log at intervals
        if iteration % self.log_interval != 0:
            return
        
        # Calculate stats
        elapsed = current_time - self.start_time
        steps_done = max(1, iteration)
        steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
        remaining_steps = self.total_iterations - iteration
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        
        progress_pct = 100 * iteration / self.total_iterations
        
        # Average losses
        avg_g = sum(self.history.g_loss) / len(self.history.g_loss) if self.history.g_loss else 0
        avg_d = sum(self.history.d_loss) / len(self.history.d_loss) if self.history.d_loss else 0
        avg_recon = sum(self.history.recon_loss) / len(self.history.recon_loss) if self.history.recon_loss else 0
        
        # Loss trends (compare last 10 with previous 10)
        def get_trend(history: deque) -> str:
            if len(history) < 20:
                return "→"
            recent = sum(list(history)[-10:]) / 10
            previous = sum(list(history)[-20:-10]) / 10
            if recent < previous * 0.95:
                return "↓"  # Improving
            elif recent > previous * 1.05:
                return "↑"  # Getting worse
            return "→"  # Stable
        
        g_trend = get_trend(self.history.g_loss)
        recon_trend = get_trend(self.history.recon_loss)
        
        # Discriminator status
        disc_status = "🟢 Active" if iteration >= self.disc_start_steps else f"⏳ Warmup ({iteration}/{self.disc_start_steps})"
        
        if self.console:
            # Build progress bar
            bar_width = 30
            filled = int(bar_width * progress_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Color-coded losses
            g_color = "green" if avg_g < 5 else "yellow" if avg_g < 10 else "red"
            recon_color = "green" if avg_recon < 0.5 else "yellow" if avg_recon < 1 else "red"
            
            # Build output
            self.console.print(
                f"[cyan][{iteration:>8,}/{self.total_iterations:,}][/cyan] "
                f"[white]{bar}[/white] "
                f"[bold]{progress_pct:5.1f}%[/bold] │ "
                f"[{g_color}]G:{avg_g:.3f}{g_trend}[/{g_color}] "
                f"D:{avg_d:.3f} "
                f"[{recon_color}]R:{avg_recon:.4f}{recon_trend}[/{recon_color}] │ "
                f"[dim]{steps_per_sec:.1f} it/s[/dim] │ "
                f"ETA: {self._format_time(eta_seconds)} │ "
                f"{disc_status}"
            )
        else:
            # Fallback plain output
            print(
                f"[{iteration:>8,}/{self.total_iterations:,}] "
                f"{progress_pct:5.1f}% | "
                f"G:{avg_g:.3f} D:{avg_d:.3f} R:{avg_recon:.4f} | "
                f"{steps_per_sec:.1f} it/s | "
                f"ETA: {self._format_time(eta_seconds)}"
            )
    
    def log_validation(self, iteration: int, metrics: Dict[str, float]):
        """Log validation results."""
        if not self.is_main:
            return
        
        if self.console:
            self.console.print()
            table = Table(title=f"🔍 Validation @ Step {iteration:,}", box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in metrics.items():
                table.add_row(key, f"{value:.6f}")
            
            self.console.print(table)
            self.console.print()
        else:
            print(f"\n=== Validation @ {iteration} ===")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
            print()
    
    def log_checkpoint(self, iteration: int, path: str):
        """Log checkpoint save."""
        if not self.is_main:
            return
        
        if self.console:
            self.console.print(
                f"[bold green]💾 Checkpoint saved:[/bold green] "
                f"[cyan]{path}[/cyan] "
                f"[dim](step {iteration:,})[/dim]"
            )
        else:
            print(f"💾 Saved checkpoint: {path}")
    
    def log_resume(self, iteration: int, path: str):
        """Log checkpoint resume."""
        if not self.is_main:
            return
        
        if self.console:
            self.console.print(
                f"[bold yellow]📂 Resumed from:[/bold yellow] "
                f"[cyan]{path}[/cyan] "
                f"[dim](step {iteration:,})[/dim]"
            )
        else:
            print(f"📂 Resumed from: {path} (step {iteration})")
    
    def log_gpu_status(self):
        """Log GPU memory usage."""
        if not self.is_main:
            return
        
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and self.console:
                self.console.print("\n[bold]GPU Status:[/bold]")
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        idx, mem_used, mem_total, util = [p.strip() for p in parts]
                        pct = int(100 * int(mem_used) / int(mem_total))
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        self.console.print(
                            f"  GPU {idx}: [{bar}] {mem_used}/{mem_total}MB ({util}%)"
                        )
                self.console.print()
        except Exception:
            pass  # Silently fail if nvidia-smi not available
    
    def log_training_complete(self, final_iteration: int):
        """Log training completion."""
        if not self.is_main:
            return
        
        total_time = time.time() - self.start_time
        
        if self.console:
            self.console.print()
            panel = Panel(
                f"[bold green]✅ Training Complete![/bold green]\n\n"
                f"[cyan]Final iteration:[/cyan] {final_iteration:,}\n"
                f"[cyan]Total time:[/cyan] {self._format_time(total_time)}\n"
                f"[cyan]Avg speed:[/cyan] {final_iteration / total_time:.1f} it/s",
                title="🎉 Success",
                border_style="green"
            )
            self.console.print(panel)
        else:
            print(f"\n{'='*60}")
            print(f"✅ Training Complete!")
            print(f"   Final iteration: {final_iteration:,}")
            print(f"   Total time: {self._format_time(total_time)}")
            print(f"{'='*60}\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d {hours}h"


# Simple progress bar fallback
class SimpleProgressBar:
    """Simple progress bar when rich is not available."""
    
    def __init__(self, total: int, initial: int = 0, desc: str = ""):
        self.total = total
        self.current = initial
        self.desc = desc
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        self.current += n
    
    def set_postfix(self, metrics: Dict[str, str]):
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        pct = 100 * self.current / self.total
        
        metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"\r[{self.current}/{self.total}] {pct:.1f}% | {rate:.1f} it/s | {metrics_str}", end="")
    
    def close(self):
        print()
