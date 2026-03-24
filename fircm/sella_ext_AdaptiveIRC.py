import time
import copy
import numpy as np
from sella.optimize.irc import IRC, IRCInnerLoopConvergenceFailure
from utils import log


class AdaptiveIRC(IRC):
    """
    An extension of Sella's IRC that automatically shrinks the step size (dx)
    upon inner loop convergence failure, grows it back when the path is stable,
    and can roll back to previously accepted IRC points when noisy PES regions
    trap the path.

    Additional behavior:
    - dx growth/shrink is rounded to a user-friendly grid (e.g. 0.005 or 0.010)
    - convergence tolerates a slightly negative minimum Hessian eigenvalue
      to handle flat minima / numerical noise
    - discarded trial steps are NOT written to trajectory
    - the printed/logged step number follows accepted trajectory points
    
    Tested with Sella 2.3.x
    Recommended base: Sella v2.3.5
    """
    def __init__(self, *args, max_dx=0.12, min_dx=0.02,
                 shrink_factor=0.5, grow_factor=1.25, grow_after=4,
                 dx_quantum=0.005, eig_tol=1e-5,
                 max_history=8, max_rollback=4,
                 rollback_factor=0.75, same_point_max_retries=1,
                 **kwargs):
        kwargs.setdefault('dx', 0.04)
        super().__init__(*args, **kwargs)
        # Store the initial dx value
        self.initial_dx = self.dx

        self.max_dx = max_dx
        self.min_dx = min_dx
        self.shrink_factor = shrink_factor
        self.grow_factor = grow_factor
        self.grow_after = grow_after
        self.dx_quantum = dx_quantum
        self.eig_tol = eig_tol

        self.max_history = max_history
        self.max_rollback = max_rollback
        self.rollback_factor = rollback_factor
        self.same_point_max_retries = same_point_max_retries

        self.consecutive_successes = 0

        # Accepted-step counter: this is what traj corresponds to
        self.accepted_steps = 0

        # History of accepted states only
        self.history = []

        # IRC.d1 is initialized later in IRC.irun()
        self.d1 = None

    def _round_dx(self, dx):
        q = self.dx_quantum
        if q is None or q <= 0:
            return dx
        return round(dx / q) * q

    def _clip_round_dx(self, dx):
        dx = min(max(dx, self.min_dx), self.max_dx)
        dx = self._round_dx(dx)
        dx = min(max(dx, self.min_dx), self.max_dx)
        return dx

    def _copy_cache_dict(self, dct):
        out = {}
        for key, val in dct.items():
            if isinstance(val, np.ndarray):
                out[key] = val.copy()
            else:
                out[key] = copy.deepcopy(val)
        return out

    def _snapshot_state(self):
        if self.pes.H.B is not None:
            H_backup = self.pes.H.B.copy()
        else:
            H_backup = None

        state = dict(
            x=self.pes.get_x().copy(),
            d1=None if self.d1 is None else self.d1.copy(),
            first=self.first,
            dx=self.dx,
            H=H_backup,
            curr=self._copy_cache_dict(self.pes.curr),
            last=self._copy_cache_dict(self.pes.last),
        )
        return state

    def _restore_state(self, state):
        self.pes.set_x(state['x'])
        self.d1 = None if state['d1'] is None else state['d1'].copy()
        self.first = state['first']
        self.dx = state['dx']
        self.pes.set_H(state['H'], initialized=(state['H'] is not None))
        self.pes.curr = self._copy_cache_dict(state['curr'])
        self.pes.last = self._copy_cache_dict(state['last'])

    def _push_history(self):
        self.history.append(self._snapshot_state())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _accepted_stepno(self):
        return self.accepted_steps + 1

    def _write_msg(self, msg, tag="IRC"):
        # Format for console using the new standard log, while keeping the original string for the logfile
        console_msg = msg.strip().replace("[AdaptiveIRC] ", "")
        log(tag, console_msg)
        if self.logfile is not None:
            self.logfile.write(msg)
            self.logfile.flush()

    def _write_accepted_traj(self):
        if self.pes.traj is not None:
            self.pes.write_traj()

    def _get_lambda_min(self):
        evals = getattr(self.pes.H, 'evals', None)
        if evals is None:
            return np.nan
        return evals[0]

    def run(self, *args, **kwargs):
        # Reset internal states every time a new run (forward/reverse) starts
        self.dx = self._clip_round_dx(self.initial_dx)
        self.consecutive_successes = 0
        self.accepted_steps = 0
        self.history = [self._snapshot_state()]

        direction = kwargs.get('direction', 'forward')
        msg = f"  [AdaptiveIRC] Starting new run (direction={direction}). Reset dx to {self.dx:.4f}\n"
        self._write_msg(msg, tag="IRC")

        return super().run(*args, **kwargs)

    def converged(self, forces=None):
        evals = getattr(self.pes.H, 'evals', None)
        if evals is None:
            return False
        return self.pes.converged(self.fmax)[0] and (evals[0] > -self.eig_tol)

    def step(self):
        same_point_retry_count = 0

        while True:
            trial_state = self._snapshot_state()

            # Disable trajectory output during speculative trial steps
            traj_backup = self.pes.traj
            self.pes.traj = None

            try:
                super().step()
            except IRCInnerLoopConvergenceFailure as e:
                self.pes.traj = traj_backup

                stepno = self._accepted_stepno()
                error_name = type(e).__name__
                error_msg = str(e) if str(e) else "No error message"
                msg = (
                    f"  [AdaptiveIRC] Step {stepno:d} failed "
                    f"({error_name}: {error_msg}) at dx={self.dx:.4f}.\n"
                )
                self._write_msg(msg, tag="Warn")

                # First, try same-point retry with smaller dx
                if same_point_retry_count < self.same_point_max_retries:
                    self._restore_state(trial_state)

                    old_dx = self.dx
                    new_dx = self._clip_round_dx(self.dx * self.shrink_factor)
                    self.consecutive_successes = 0
                    same_point_retry_count += 1

                    if new_dx < old_dx:
                        self.dx = new_dx
                        retry_msg = (
                            f"  [AdaptiveIRC] Step {stepno:d}: restored same point. "
                            f"Retrying with smaller dx={self.dx:.4f} ...\n"
                        )
                        self._write_msg(retry_msg, tag="IRC")
                        continue

                # Next, roll back to older accepted points.
                # Drop the current accepted state from history so that history[-1]
                # always means the latest still-valid safe point.
                if len(self.history) > 1 and self.max_rollback > 0:
                    history_len_before = len(self.history)
                    oldest_allowed_len = max(1, history_len_before - self.max_rollback)

                    if len(self.history) > oldest_allowed_len:
                        current_state = self.history.pop()
                        hist_state = self.history[-1]

                        self._restore_state(hist_state)

                        old_dx = current_state['dx']
                        target_dx = min(hist_state['dx'], trial_state['dx'] * self.rollback_factor)
                        new_dx = self._clip_round_dx(target_dx)
                        self.consecutive_successes = 0
                        same_point_retry_count = 0
                        self.dx = new_dx

                        rollback_steps = history_len_before - len(self.history)
                        rb_msg = (
                            f"  [AdaptiveIRC] Step {stepno:d}: rolling back {rollback_steps:d} "
                            f"accepted step(s); dx {old_dx:.4f} -> {self.dx:.4f}.\n"
                        )
                        self._write_msg(rb_msg, tag="IRC")
                        continue

                # No more rollback options: only succeed if convergence is genuinely satisfied
                self._restore_state(trial_state)

                if self.converged():
                    lam_min = self._get_lambda_min()
                    msg_conv = (
                        f"  [AdaptiveIRC] Step {stepno:d}: rollback budget exhausted, "
                        f"but convergence criteria are satisfied "
                        f"(lambda_min={lam_min:.6e}, eig_tol={self.eig_tol:.6e}). Stopping.\n"
                    )
                    self._write_msg(msg_conv, tag="IRC")
                    break

                try:
                    current_fmax = np.linalg.norm(
                        self.pes.get_projected_forces(), axis=1
                    ).max()
                except Exception:
                    current_fmax = np.nan

                lam_min = self._get_lambda_min()
                abort_msg = (
                    f"AdaptiveIRC aborted at planned step {stepno:d}: "
                    f"rollback budget exhausted, fmax={current_fmax:.4f}, "
                    f"lambda_min={lam_min:.6e}, eig_tol={self.eig_tol:.6e}, "
                    f"dx={self.dx:.4f}.\n"
                )
                self._write_msg(f"  [AdaptiveIRC] {abort_msg}", tag="Fail")
                raise RuntimeError(abort_msg.strip()) from e

            else:
                self.pes.traj = traj_backup

                # Successful accepted outer step: now write a single traj frame
                self._write_accepted_traj()
                self.accepted_steps += 1
                self._push_history()

                self.consecutive_successes += 1
                if self.consecutive_successes >= self.grow_after:
                    old_dx = self.dx
                    new_dx = self._clip_round_dx(self.dx * self.grow_factor)
                    if new_dx > old_dx:
                        self.dx = new_dx
                        msg = (
                            f"  [AdaptiveIRC] Step {self.accepted_steps:d}: path looks stable. "
                            f"Increasing dx: {old_dx:.4f} -> {self.dx:.4f}\n"
                        )
                        self._write_msg(msg, tag="IRC")
                    self.consecutive_successes = 0

                break

    def log(self, forces=None):
        if self.logfile is None:
            return

        try:
            _, fmax, cmax = self.pes.converged(self.fmax)
        except Exception:
            fmax = np.nan
            cmax = np.nan

        e = self.pes.get_f()
        T = time.strftime("%H:%M:%S", time.localtime())
        name = self.__class__.__name__
        lam_min = self._get_lambda_min()

        if self.nsteps == 0:
            self.logfile.write(
                f"{' ' * len(name)} {'Step':>4s} {'Time':>8s} {'Energy':>15s} "
                f"{'fmax':>12s} {'cmax':>12s} {'dx':>10s} {'lam_min':>14s}\n"
            )

        self.logfile.write(
            f"{name} {self._accepted_stepno():>3d} {T:>8s} {e:>15.6f} "
            f"{fmax:>12.4f} {cmax:>12.4f} {self.dx:>10.4f} {lam_min:>14.6e}\n"
        )
        self.logfile.flush()
