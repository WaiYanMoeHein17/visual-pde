"""
ADER-DG Temporal Convergence Test

Measures the temporal convergence rate of the SSPRK/RK4 timestepping scheme
(as implemented in ader_dg.js) and compares with classical Runge-Kutta methods.

Orders 1-3 use SSPRK methods in Shu-Osher form:
  u^(i) = alpha * u^n + delta * u^(i-1) + beta * dt * F(u^(i-1))

Order 4 uses classical RK4 with complete state storage:
  Stages 1-3 produce complete states y2, y3, y4.
  Final: u^{n+1} = (-u^n + y2 + 2*y3 + y4)/3 + (dt/6)*F(y4)

Usage:
  python _numerical-methods/convergence_test.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ============================================================
# Time integrators
# ============================================================

def ader_step(u, rhs, dt, order):
    """SSPRK/RK4 timestepping, matching ader_dg.js exactly."""
    if order == 1:
        # Forward Euler
        return u + dt * rhs(u)
    elif order == 2:
        # SSPRK2 (Shu-Osher form)
        y1 = u + dt * rhs(u)
        return 0.5 * u + 0.5 * y1 + 0.5 * dt * rhs(y1)
    elif order == 3:
        # SSPRK3 (Shu-Osher form)
        y1 = u + dt * rhs(u)
        y2 = (3.0 / 4.0) * u + (1.0 / 4.0) * y1 + (1.0 / 4.0) * dt * rhs(y1)
        return (1.0 / 3.0) * u + (2.0 / 3.0) * y2 + (2.0 / 3.0) * dt * rhs(y2)
    else:
        # Classical RK4 with complete state storage
        y2 = u + 0.5 * dt * rhs(u)
        y3 = u + 0.5 * dt * rhs(y2)
        y4 = u + dt * rhs(y3)
        return (-u + y2 + 2.0 * y3 + y4) / 3.0 + (dt / 6.0) * rhs(y4)


def rk1_step(u, rhs, dt):
    """Forward Euler (reference)."""
    return u + dt * rhs(u)


def rk2_step(u, rhs, dt):
    """Explicit midpoint (reference)."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    return u + dt * k2


def rk4_step(u, rhs, dt):
    """Classical RK4 (reference)."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(stepper, u0, rhs, dt, T):
    """Integrate from t=0 to t=T using the given stepper."""
    u = u0.copy()
    t = 0.0
    while t < T - 1e-14:
        step = min(dt, T - t)
        u = stepper(u, rhs, step)
        t += step
    return u


# ============================================================
# Test problems
# ============================================================

def exponential_decay():
    """du/dt = -u, u(0) = 1. Exact: exp(-t)."""
    def rhs(u):
        return -u
    u0 = np.array([1.0])
    def exact(t):
        return np.array([np.exp(-t)])
    return "Exponential Decay", rhs, u0, exact


def harmonic_oscillator():
    """du/dt = v, dv/dt = -u. Exact: u=cos(t), v=-sin(t)."""
    def rhs(u):
        return np.array([u[1], -u[0]])
    u0 = np.array([1.0, 0.0])
    def exact(t):
        return np.array([np.cos(t), -np.sin(t)])
    return "Harmonic Oscillator", rhs, u0, exact


# ============================================================
# 1D advection with DG-like element structure
# ============================================================

class DGAdvection1D:
    """
    Linear advection u_t + u_x = 0 on [0, 2pi] periodic.

    Spatial discretization uses central differences with DG element structure:
    - Elements of (order+1) nodes each
    - Boundary nodes are duplicated (rightmost of elem i = leftmost of elem i+1)
    - dgStepLeft/dgStepRight skip duplicate boundary nodes
    - After each timestep, shared boundary nodes are averaged

    This matches the GPU implementation in simulation_shaders.js.
    """

    def __init__(self, n_elem, order):
        self.order = order
        self.nodes_per_elem = order + 1
        self.n_elem = n_elem
        self.n_dof = n_elem * self.nodes_per_elem
        self.L = 2.0 * np.pi
        # dx is the spacing between adjacent nodes (including within elements)
        # Total unique positions = n_elem * order (boundary nodes overlap)
        self.dx = self.L / (n_elem * order)
        # Physical coordinates for each DOF
        self.x = np.zeros(self.n_dof)
        for e in range(n_elem):
            for i in range(self.nodes_per_elem):
                idx = e * self.nodes_per_elem + i
                self.x[idx] = (e * order + i) * self.dx

    def _step_left(self, idx):
        """dgStepLeft: skip duplicate boundary node when at left edge of element."""
        npe = self.nodes_per_elem
        local = idx % npe
        if local == 0:
            prev_elem = (idx // npe) - 1
            raw = prev_elem * npe + (self.order - 1)
            return raw % self.n_dof
        return (idx - 1) % self.n_dof

    def _step_right(self, idx):
        """dgStepRight: skip duplicate boundary node when at right edge of element."""
        npe = self.nodes_per_elem
        local = idx % npe
        if local == self.order:
            next_elem = (idx // npe) + 1
            raw = next_elem * npe + 1
            return raw % self.n_dof
        return (idx + 1) % self.n_dof

    def rhs(self, u):
        """Spatial RHS: du/dt = -du/dx using central differences."""
        f = np.zeros_like(u)
        for i in range(self.n_dof):
            iL = self._step_left(i)
            iR = self._step_right(i)
            # Central difference: -u_x = -(u_R - u_L) / (2*dx)
            f[i] = -(u[iR] - u[iL]) / (2.0 * self.dx)
        return f

    def boundary_avg(self, u):
        """Average duplicated boundary nodes (matches RDShaderDGBoundaryAvg)."""
        result = u.copy()
        npe = self.nodes_per_elem
        for e in range(self.n_elem):
            e_next = (e + 1) % self.n_elem
            right = e * npe + self.order
            left = e_next * npe
            avg = 0.5 * (u[right] + u[left])
            result[right] = avg
            result[left] = avg
        return result

    def initial_condition(self):
        return np.sin(self.x)

    def exact_solution(self, t):
        return np.sin(self.x - t)

    def l2_error(self, u, u_exact):
        return np.sqrt(np.mean((u - u_exact) ** 2))

    def linf_error(self, u, u_exact):
        return np.max(np.abs(u - u_exact))


# ============================================================
# Convergence study
# ============================================================

def convergence_study_ode(problem_fn, dt_values, T, methods):
    """
    Run convergence study for an ODE problem.
    Returns (name, l2_results, linf_results) where each is
    dict: method_name -> list of (dt, error) pairs.
    """
    name, rhs, u0, exact = problem_fn()
    u_exact = exact(T)
    l2_results = {}
    linf_results = {}

    for method_name, stepper in methods:
        l2_errors = []
        linf_errors = []
        for dt in dt_values:
            u_final = integrate(stepper, u0, rhs, dt, T)
            diff = u_final - u_exact
            l2_errors.append(np.sqrt(np.mean(diff ** 2)))
            linf_errors.append(np.max(np.abs(diff)))
        l2_results[method_name] = list(zip(dt_values, l2_errors))
        linf_results[method_name] = list(zip(dt_values, linf_errors))

    return name, l2_results, linf_results


def convergence_study_advection_h(order_values, n_elem_values, cfl_frac, T):
    """
    h-refinement convergence study for 1D advection with DG structure.
    Varies the number of elements (and proportionally dt via CFL) to test
    the combined spatial + temporal convergence rate.

    With central differences (O(dx²)) and CFL-limited dt ~ dx, refining dt
    alone just hits the spatial error floor. h-refinement tests both together.

    Returns (l2_results, linf_results) where each is
    dict: label -> list of (dx, error) pairs.
    """
    l2_results = {}
    linf_results = {}

    for order in order_values:
        l2_dx_err = []
        linf_dx_err = []
        for n_elem in n_elem_values:
            dg = DGAdvection1D(n_elem, order)
            dt = cfl_frac * dg.dx / (2 * order + 1)
            u0 = dg.initial_condition()
            u_exact = dg.exact_solution(T)
            rhs_fn = dg.rhs

            u = u0.copy()
            t = 0.0
            while t < T - 1e-14:
                step = min(dt, T - t)
                u = ader_step(u, rhs_fn, step, order)
                u = dg.boundary_avg(u)
                t += step
            l2_dx_err.append((dg.dx, dg.l2_error(u, u_exact)))
            linf_dx_err.append((dg.dx, dg.linf_error(u, u_exact)))

        label = f"ADER {order}"
        l2_results[label] = l2_dx_err
        linf_results[label] = linf_dx_err

    return l2_results, linf_results


def compute_rates(dt_err_pairs):
    """Compute convergence rates from (dt, error) pairs."""
    rates = [None]
    for i in range(1, len(dt_err_pairs)):
        dt_prev, err_prev = dt_err_pairs[i - 1]
        dt_curr, err_curr = dt_err_pairs[i]
        if err_curr > 0 and err_prev > 0:
            rates.append(np.log2(err_prev / err_curr))
        else:
            rates.append(None)
    return rates


# ============================================================
# Output formatting
# ============================================================

def format_table(name, results, expected_rates=None, x_label="dt", norm_label=""):
    """Format convergence table as text."""
    lines = []
    for method_name, dt_err in results.items():
        rates = compute_rates(dt_err)
        suffix = f" [{norm_label}]" if norm_label else ""
        lines.append(f"\n--- {name}: {method_name}{suffix} ---")
        lines.append(f"  {x_label:>12s}  {'error':>12s}  {'rate':>6s}")
        valid_rates = []
        for i, (dt, err) in enumerate(dt_err):
            rate_str = f"{rates[i]:.2f}" if rates[i] is not None else "—"
            if rates[i] is not None:
                valid_rates.append(rates[i])
            lines.append(f"  {dt:12.4e}  {err:12.4e}  {rate_str:>6s}")
        if valid_rates:
            avg_rate = np.mean(valid_rates)
            exp_str = ""
            if expected_rates and method_name in expected_rates:
                exp_str = f" | Expected: {expected_rates[method_name]}"
            lines.append(f"  Average rate: {avg_rate:.2f}{exp_str}")
    return "\n".join(lines)


def format_summary(all_results, expected_rates, norm_label=""):
    """Format summary comparison table."""
    header = f"SUMMARY [{norm_label}]" if norm_label else "SUMMARY"
    lines = ["\n" + "=" * 60, header, "=" * 60]
    lines.append(f"  {'Method':<20s}  {'Avg Rate':>10s}  {'Expected':>10s}")
    lines.append("  " + "-" * 44)
    for method_name, dt_err in all_results.items():
        rates = compute_rates(dt_err)
        valid = [r for r in rates if r is not None]
        avg = np.mean(valid) if valid else float("nan")
        exp = expected_rates.get(method_name, "?")
        lines.append(f"  {method_name:<20s}  {avg:10.2f}  {str(exp):>10s}")
    return "\n".join(lines)


# ============================================================
# Plotting
# ============================================================

def plot_convergence(title, results, expected_rates, filename, x_label="dt", y_label="Error"):
    """Create log-log convergence plot with reference slopes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

    for i, (method_name, xy_err) in enumerate(results.items()):
        xs = [x for x, _ in xy_err]
        errs = [e for _, e in xy_err]
        ax.loglog(
            xs, errs,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=method_name,
            linewidth=1.5,
            markersize=5,
        )

    # Reference slope lines
    all_xs = []
    for xy_err in results.values():
        all_xs.extend([x for x, _ in xy_err])
    x_min, x_max = min(all_xs), max(all_xs)
    x_ref = np.array([x_min, x_max])

    unit = "h" if x_label == "dx" else x_label
    for rate, style, label in [(1, "--", f"O({unit})"), (2, "-.", f"O({unit}²)"), (3, ":", f"O({unit}³)"), (4, ":", f"O({unit}⁴)")]:
        if rate in [expected_rates.get(k) for k in expected_rates]:
            mid_err = np.median([e for xy_err in results.values() for _, e in xy_err])
            mid_x = np.sqrt(x_min * x_max)
            c = mid_err / mid_x ** rate
            ax.loglog(
                x_ref, c * x_ref ** rate,
                style, color="gray", alpha=0.5, label=label,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text_lines = ["ADER-DG Temporal Convergence Test", "=" * 40, ""]

    T = 1.0  # Integration time

    # ----------------------------------------------------------
    # ODE Tests
    # ----------------------------------------------------------
    dt_base = 0.1
    dt_values = [dt_base / (2 ** k) for k in range(6)]

    # Methods to test
    ode_methods = [
        ("ADER 1", lambda u, rhs, dt: ader_step(u, rhs, dt, 1)),
        ("ADER 2", lambda u, rhs, dt: ader_step(u, rhs, dt, 2)),
        ("ADER 3", lambda u, rhs, dt: ader_step(u, rhs, dt, 3)),
        ("ADER 4", lambda u, rhs, dt: ader_step(u, rhs, dt, 4)),
        ("RK1 (Euler)", rk1_step),
        ("RK2 (Midpoint)", rk2_step),
        ("RK4", rk4_step),
    ]

    expected_ode = {
        "ADER 1": 1, "ADER 2": 2, "ADER 3": 3, "ADER 4": 4,
        "RK1 (Euler)": 1, "RK2 (Midpoint)": 2, "RK4": 4,
    }

    all_ode_results = {}

    for problem_fn in [exponential_decay, harmonic_oscillator]:
        name, l2_results, linf_results = convergence_study_ode(problem_fn, dt_values, T, ode_methods)
        text_lines.append(format_table(name, l2_results, expected_ode, norm_label="L2"))
        text_lines.append(format_table(name, linf_results, expected_ode, norm_label="L∞"))

    # Summary for ODE tests (merge both problems' results by method)
    merged_l2 = {}
    merged_linf = {}
    for problem_fn in [exponential_decay, harmonic_oscillator]:
        _, l2_results, linf_results = convergence_study_ode(problem_fn, dt_values, T, ode_methods)
        for k, v in l2_results.items():
            if k not in merged_l2:
                merged_l2[k] = v
        for k, v in linf_results.items():
            if k not in merged_linf:
                merged_linf[k] = v

    text_lines.append(format_summary(merged_l2, expected_ode, norm_label="L2"))
    text_lines.append(format_summary(merged_linf, expected_ode, norm_label="L∞"))

    # Plot ODE results (2x2: rows = L2/L∞, cols = problems)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "p", "h"]

    for col_idx, problem_fn in enumerate([exponential_decay, harmonic_oscillator]):
        name, l2_results, linf_results = convergence_study_ode(problem_fn, dt_values, T, ode_methods)

        for row_idx, (results, norm_label) in enumerate([(l2_results, "L₂"), (linf_results, "L∞")]):
            ax = axes[row_idx, col_idx]

            for i, (method_name, dt_err) in enumerate(results.items()):
                dts = [d for d, _ in dt_err]
                errs = [e for _, e in dt_err]
                ax.loglog(
                    dts, errs,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=method_name,
                    linewidth=1.5, markersize=5,
                )

            # Reference slopes
            dt_ref = np.array([dt_values[-1], dt_values[0]])
            for rate, style, label in [(1, "--", "O(dt)"), (2, "-.", "O(dt²)"), (3, ":", "O(dt³)"), (4, ":", "O(dt⁴)")]:
                mid_err = np.median([e for dt_err in results.values() for _, e in dt_err])
                mid_dt = np.sqrt(dt_values[0] * dt_values[-1])
                c = mid_err / mid_dt ** rate
                ax.loglog(dt_ref, c * dt_ref ** rate, style, color="gray", alpha=0.5, label=label)

            ax.set_xlabel("dt")
            ax.set_ylabel(f"Error ({norm_label})")
            ax.set_title(f"{name} ({norm_label})")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("ODE Temporal Convergence", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "convergence-ode.png"), dpi=150)
    plt.close(fig)
    print("  Saved convergence-ode.png")

    # ----------------------------------------------------------
    # Advection Test (h-refinement)
    # ----------------------------------------------------------
    # With central differences (O(dx²)) and CFL-limited dt ~ dx, refining dt
    # alone just hits the spatial error floor. h-refinement tests the combined
    # spatial + temporal convergence by varying mesh size with dt = CFL * dx.
    text_lines.append("\n\n" + "=" * 60)
    text_lines.append("1D ADVECTION — h-REFINEMENT (DG elements, dt ~ CFL * dx)")
    text_lines.append("=" * 60)
    text_lines.append("Central difference spatial operator (O(dx²)).")
    text_lines.append("CFL fraction = 0.5. Convergence rate reflects min(temporal, spatial) order.")

    order_values = [1, 2, 3, 4]
    n_elem_values = [16, 32, 64, 128, 256, 512]
    cfl_frac = 0.5

    adv_l2, adv_linf = convergence_study_advection_h(order_values, n_elem_values, cfl_frac, T)

    # Expected combined rate: min(temporal_order, spatial_order=2)
    # ADER 1: min(1, 2) = 1.  ADER 2-4: min(2, 2) = 2.
    expected_adv = {f"ADER {p}": min(p, 2) for p in order_values}
    text_lines.append(format_table("1D Advection h-refine", adv_l2, expected_adv, x_label="dx", norm_label="L2"))
    text_lines.append(format_table("1D Advection h-refine", adv_linf, expected_adv, x_label="dx", norm_label="L∞"))
    text_lines.append(format_summary(adv_l2, expected_adv, norm_label="L2"))
    text_lines.append(format_summary(adv_linf, expected_adv, norm_label="L∞"))

    # Plot advection results (L2 and L∞)
    plot_convergence(
        "1D Advection — h-Refinement (L₂)",
        adv_l2, expected_adv, "convergence-advection-l2.png", x_label="dx", y_label="Error (L₂)",
    )
    plot_convergence(
        "1D Advection — h-Refinement (L∞)",
        adv_linf, expected_adv, "convergence-advection-linf.png", x_label="dx", y_label="Error (L∞)",
    )

    # ----------------------------------------------------------
    # Write text output
    # ----------------------------------------------------------
    text_path = os.path.join(OUTPUT_DIR, "convergence-results.txt")
    with open(text_path, "w") as f:
        f.write("\n".join(text_lines))
    print(f"  Saved convergence-results.txt")
    print("\nDone. Results in: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()
