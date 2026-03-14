import time
import numpy as np
from sella.optimize.irc import IRC, IRCInnerLoopConvergenceFailure

class AdaptiveIRC(IRC):
    """
    An extension of Sella's IRC that automatically shrinks the step size (dx) 
    upon inner loop convergence failure (or ANY numerical math error), 
    and grows it back when the path is stable.
    """
    def __init__(self, *args, max_dx=0.16, min_dx=0.01, 
                 shrink_factor=0.5, grow_factor=2.0, grow_after=3, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the initial dx value
        self.initial_dx = self.dx
        
        self.max_dx = max_dx
        self.min_dx = min_dx
        self.shrink_factor = shrink_factor
        self.grow_factor = grow_factor
        self.grow_after = grow_after
        self.consecutive_successes = 0
        
        # Add: Force convergence flag
        self._force_converged = False

    def run(self, *args, **kwargs):
        # Reset internal states every time a new run (forward/reverse) starts
        self.dx = self.initial_dx
        self.consecutive_successes = 0
        self._force_converged = False
        
        direction = kwargs.get('direction', 'forward')
        msg = f"  [AdaptiveIRC] Starting new run (direction={direction}). Reset dx to {self.dx:.4f}\n"
        print(msg, end="")
        if self.logfile is not None:
            self.logfile.write(msg)
            self.logfile.flush()
            
        return super().run(*args, **kwargs)

    def converged(self, forces=None):
        # Terminate the ASE loop upon Graceful Exit
        if self._force_converged:
            return True
        return super().converged(forces)

    def step(self):
        # Do nothing if already forcefully converged (fail-safe)
        if self._force_converged:
            return

        # Backup state
        x_backup = self.pes.get_x().copy()
        d1_backup = self.d1.copy()
        first_backup = self.first
        
        if self.pes.H.B is not None:
            H_backup = self.pes.H.B.copy()
        else:
            H_backup = None

        while True:
            try:
                super().step()
                
                # --- On Success ---
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
                break
                
            # Catch ALL exceptions
            except Exception as e:
                error_name = type(e).__name__
                error_msg = str(e) if str(e) else "No error message"
                
                # --- On Failure ---
                msg = f"  [AdaptiveIRC] Step failed ({error_name}: {error_msg}) at dx={self.dx:.4f}.\n"
                print(msg, end="")
                if self.logfile is not None:
                    self.logfile.write(msg)
                
                # Restore state
                self.pes.set_x(x_backup)
                self.d1 = d1_backup.copy()
                self.first = first_backup
                self.pes.set_H(H_backup, initialized=(H_backup is not None))
                
                # Shrink step size
                self.dx *= self.shrink_factor
                self.consecutive_successes = 0
                
                if self.dx < self.min_dx:
                    # Graceful Exit Logic
                    try:
                        current_fmax = np.linalg.norm(self.pes.get_projected_forces(), axis=1).max()
                    except Exception:
                        current_fmax = 1.0  
                    
                    target_fmax = getattr(self, 'fmax', 0.01)
                    
                    if current_fmax < target_fmax * 2.5:
                        msg_grace = f"  [AdaptiveIRC] Reached min_dx, but fmax ({current_fmax:.4f}) is small enough. Assuming convergence.\n"
                        print(msg_grace, end="")
                        if self.logfile is not None:
                            self.logfile.write(msg_grace)
                            self.logfile.flush()
                        
                        # Set a flag to signal convergence to ASE
                        self._force_converged = True
                        break
                    else:
                        abort_msg = (
                            f"AdaptiveIRC aborted: step size (dx) shrank below min_dx ({self.min_dx}) "
                            f"and fmax ({current_fmax:.4f}) is still high.\n"
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
        if self.logfile is None:
            return
        try:
            fmax = np.linalg.norm(self.pes.get_projected_forces(), axis=1).max()
        except Exception:
            fmax = 0.0
            
        e = self.pes.get_f()
        T = time.strftime("%H:%M:%S", time.localtime())
        name = self.__class__.__name__
        
        if self.nsteps == 0:
            self.logfile.write(f"{' ' * len(name)} {'Step':>4s} {'Time':>8s} {'Energy':>15s} {'fmax':>12s} {'dx':>10s}\n")
            
        self.logfile.write(f"{name} {self.nsteps:>3d} {T:>8s} {e:>15.6f} {fmax:>12.4f} {self.dx:>10.4f}\n")
        self.logfile.flush()
