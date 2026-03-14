import time
import numpy as np
from sella.optimize.irc import IRC, IRCInnerLoopConvergenceFailure

class AdaptiveIRC(IRC):
    """
    An extension of Sella's IRC that automatically shrinks the step size (dx) 
    upon inner loop convergence failure, and grows it back when the path is stable.
    """
    def __init__(self, *args, max_dx=0.12, min_dx=0.02, 
                 shrink_factor=0.5, grow_factor=2.0, grow_after=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dx = max_dx
        self.min_dx = min_dx
        self.shrink_factor = shrink_factor
        self.grow_factor = grow_factor
        self.grow_after = grow_after
        self.consecutive_successes = 0

    def step(self):
        # --- Backup state before attempting a step ---
        # Backup atomic positions and dummy positions
        x_backup = self.pes.get_x().copy()
        
        # Backup IRC-specific internal variables
        d1_backup = self.d1.copy()
        first_backup = self.first
        
        # Backup the PES Hessian matrix to prevent corruption on failure
        # (This is an O(N^2) copy of a small matrix, overhead is virtually zero)
        if self.pes.H.B is not None:
            H_backup = self.pes.H.B.copy()
        else:
            H_backup = None

        # Loop to retry with shrinking dx until successful
        while True:
            try:
                # Execute the original Sella IRC step
                super().step()
                
                # --- On Success: Step size expansion logic ---
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.grow_after:
                    old_dx = self.dx
                    self.dx = min(self.dx * self.grow_factor, self.max_dx)
                    if self.dx > old_dx:
                        msg = f"  [AdaptiveIRC] Path looks stable. Increasing dx: {old_dx:.4f} -> {self.dx:.4f}\n"
                        print(msg, end="")
                        if self.logfile is not None:
                            self.logfile.write(msg)
                            self.logfile.flush()
                    self.consecutive_successes = 0
                
                # Step succeeded, break the retry loop
                break
                
            except IRCInnerLoopConvergenceFailure as e:
                # --- On Failure: Restore state and shrink step size ---
                msg = f"  [AdaptiveIRC] Inner loop failed at dx={self.dx:.4f}.\n"
                print(msg, end="")
                if self.logfile is not None:
                    self.logfile.write(msg)
                
                # Restore the state completely
                self.pes.set_x(x_backup)
                self.d1 = d1_backup.copy()
                self.first = first_backup
                self.pes.set_H(H_backup, initialized=(H_backup is not None))
                
                # Shrink the step size
                self.dx *= self.shrink_factor
                self.consecutive_successes = 0
                
                if self.dx < self.min_dx:
                    abort_msg = (
                        f"AdaptiveIRC aborted: step size (dx) shrank below min_dx ({self.min_dx}). "
                        "The minimum might be too shallow or the PES is discontinuous.\n"
                    )
                    if self.logfile is not None:
                        self.logfile.write(abort_msg)
                        self.logfile.flush()
                    raise RuntimeError(abort_msg.strip()) from e
                
                retry_msg = f"  [AdaptiveIRC] Restored geometry. Retrying with smaller dx={self.dx:.4f} ...\n"
                print(retry_msg, end="")
                if self.logfile is not None:
                    self.logfile.write(retry_msg)
                    self.logfile.flush()

    def log(self, forces=None):
        """Override log to include current dx value."""
        if self.logfile is None:
            return
        
        # Calculate max force on the free subspace
        try:
            fmax = np.linalg.norm(self.pes.get_projected_forces(), axis=1).max()
        except Exception:
            fmax = 0.0
            
        e = self.pes.get_f()
        T = time.strftime("%H:%M:%S", time.localtime())
        name = self.__class__.__name__
        
        # Header formatting
        if self.nsteps == 0:
            self.logfile.write(f"{' ' * len(name)} {'Step':>4s} {'Time':>8s} {'Energy':>15s} {'fmax':>12s} {'dx':>10s}\n")
            
        # Data formatting
        self.logfile.write(f"{name} {self.nsteps:>3d} {T:>8s} {e:>15.6f} {fmax:>12.4f} {self.dx:>10.4f}\n")
        self.logfile.flush()
