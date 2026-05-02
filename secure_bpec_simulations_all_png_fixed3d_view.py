# -*- coding: utf-8 -*-
"""
This script merges the three scripts:
1. single_user_rate_region_eval_v1.py
2. feedback_region_comparison.py
3. single_feedback_parameter_sweep.py

generates:
- Fig. 1: single-user feedback secure achievable region
- Fig. 2: single-user feedback vs full-feedback vs no-feedback secure comparison
- Fig. 3: full-feedback secure capacity vs non-secure feedback capacity
- Fig. 4: delta2 sweep of R2,max, with delta1 fixed
- Fig. 5: 3D surface of the single-user feedback achievable boundary
- Fig. 6: delta1 sweep of R2,max, with delta2 fixed

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 0. Parameter validation

def validate_joint_erasure(delta1, delta2, delta12):
    """
    Validate a two-user BPEC joint erasure model

    delta1  = P(Rx1 erased)
    delta2  = P(Rx2 erased)
    delta12 = P(Rx1 erased and Rx2 erased)
    """
    if not (0 <= delta1 <= 1 and 0 <= delta2 <= 1 and 0 <= delta12 <= 1):
        raise ValueError("All erasure probabilities must lie in [0,1].")

    if delta12 > min(delta1, delta2):
        raise ValueError("Need delta12 <= min(delta1, delta2).")

    if delta12 < max(0.0, delta1 + delta2 - 1.0):
        raise ValueError("Need delta12 >= max(0, delta1 + delta2 - 1).")


def check_non_degenerate(values, name="parameters", eps=1e-12):
    """Avoid zero denominators."""
    if any(abs(x) < eps for x in values):
        raise ValueError(f"Degenerate {name}: zero denominator encountered.")


# 1. Region constraints

def single_feedback_constraints(delta1, delta2, delta12):
    """
    Single-user feedback secure achievable region

    Returns constraints in the form:
        a_i R1 + b_i R2 <= 1,  i = 1,2,3
    Rx1 is the feedback receiver and Rx2 is the non-feedback receiver
    """
    validate_joint_erasure(delta1, delta2, delta12)
    check_non_degenerate(
        [
            delta1 - delta12,
            delta2 - delta12,
            1.0 - delta1,
            1.0 - delta2,
            1.0 - delta12,
        ],
        name="single-feedback parameters",
    )

    a1 = 1.0 / (1.0 - delta12)
    b1 = (
        1.0 / (delta1 - delta12)
        + delta1 * (delta2 - delta12) / ((delta1 - delta12) * (1.0 - delta2))
    )

    a2 = (
        1.0 / (1.0 - delta12)
        + (1.0 - delta2) / ((1.0 - delta12) * (delta2 - delta12))
    )
    b2 = delta1 * (1.0 - delta12) / ((delta1 - delta12) * (1.0 - delta2))

    a3 = (
        1.0 / (1.0 - delta1)
        + (1.0 - delta2) / ((1.0 - delta12) * (delta2 - delta12))
    )
    b3 = delta1 / (delta1 - delta12)

    return np.array([[a1, b1], [a2, b2], [a3, b3]], dtype=float)


def full_feedback_secure_constraints(delta1, delta2):
    """
    Full-feedback secure capacity region for independent two-user BPEC

    Returns constraints in the form:
        a_i R1 + b_i R2 <= 1,  i = 1,2

    Assumes delta12 = delta1 * delta2
    """
    dprod = delta1 * delta2

    check_non_degenerate(
        [
            delta1,
            delta2,
            1.0 - delta1,
            1.0 - delta2,
            1.0 - dprod,
        ],
        name="full-feedback secure parameters",
    )

    a1 = (
        (1.0 - delta2)
        / (delta2 * (1.0 - delta1) * (1.0 - dprod))
        + 1.0 / (1.0 - delta1)
    )
    b1 = 1.0 / (1.0 - dprod)

    a2 = 1.0 / (1.0 - dprod)
    b2 = (
        (1.0 - delta1)
        / (delta1 * (1.0 - delta2) * (1.0 - dprod))
        + 1.0 / (1.0 - delta2)
    )

    return np.array([[a1, b1], [a2, b2]], dtype=float)


def nonsecure_feedback_constraints(delta1, delta2, delta12):
    """
    Non-secure feedback capacity benchmark for the two-user BPEC

    Returns constraints in the form:
        a_i R1 + b_i R2 <= 1,  i = 1,2
    """
    validate_joint_erasure(delta1, delta2, delta12)
    check_non_degenerate(
        [
            1.0 - delta1,
            1.0 - delta2,
            1.0 - delta12,
        ],
        name="non-secure feedback parameters",
    )

    a1 = 1.0 / (1.0 - delta1)
    b1 = 1.0 / (1.0 - delta12)

    a2 = 1.0 / (1.0 - delta12)
    b2 = 1.0 / (1.0 - delta2)

    return np.array([[a1, b1], [a2, b2]], dtype=float)


def no_feedback_secure_baseline(delta1, delta2):
    """
    No-feedback secrecy baseline

    Only the stronger receiver has positive individual secrecy rate
      if delta1 < delta2: User 1 is stronger, R1 <= delta2 - delta1 and R2 = 0.
      if delta2 < delta1: User 2 is stronger, R2 <= delta1 - delta2 and R1 = 0.
      if delta1 = delta2: only the origin

    Used only as a baseline visualization
    """
    if delta1 < delta2:
        return {"axis": "R1", "max_rate": delta2 - delta1}
    if delta2 < delta1:
        return {"axis": "R2", "max_rate": delta1 - delta2}
    return {"axis": "origin", "max_rate": 0.0}


def no_feedback_r2max(delta1, delta2):
    """
    No-feedback individual secrecy rate for User 2
    User 2 has positive no-feedback secrecy rate only if it is stronger:
        R2,max = max(delta1 - delta2, 0)
    """
    return max(delta1 - delta2, 0.0)


# 2. Generic region utilities

def boundary_from_constraints(constraints, R1_grid=None, num=2000):
    """
    Compute the upper boundary:
        R2_max(R1) = min_i (1 - a_i R1) / b_i
    for constraints a_i R1 + b_i R2 <= 
    """
    R1_axis_max = min(1.0 / a for a, _ in constraints)

    if R1_grid is None:
        R1_grid = np.linspace(0.0, R1_axis_max, num)

    bounds = []
    for a, b in constraints:
        bounds.append((1.0 - a * R1_grid) / b)

    R2_boundary = np.minimum.reduce(bounds)
    R2_boundary = np.where(R2_boundary >= 0, R2_boundary, np.nan)
    return R1_grid, R2_boundary


def region_summary(constraints):
    """Return R1_max, R2_max, and R_sym for a linear rate region."""
    R1_max = min(1.0 / a for a, _ in constraints)
    R2_max = min(1.0 / b for _, b in constraints)
    R_sym = min(1.0 / (a + b) for a, b in constraints)
    return R1_max, R2_max, R_sym


def r1max_from_constraints(constraints):
    return min(1.0 / a for a, _ in constraints)


def r2max_from_constraints(constraints):
    return min(1.0 / b for _, b in constraints)


def rsym_from_constraints(constraints):
    return min(1.0 / (a + b) for a, b in constraints)


# 3. Figure 1: Single-user feedback secure region

def plot_single_feedback_region(
    delta1=0.4,
    delta2=0.6,
    delta12=None,
    save_dir="figures",
    filename_base="single_feedback_secure_region",
):
    """
    Plot the single-user feedback secure achievable region
    """
    if delta12 is None:
        delta12 = delta1 * delta2

    constraints = single_feedback_constraints(delta1, delta2, delta12)
    R1_boundary, R2_boundary = boundary_from_constraints(constraints)
    R1_max, R2_max, R_sym = region_summary(constraints)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.fill_between(
        R1_boundary,
        0,
        R2_boundary,
        alpha=0.35,
    )
    ax.plot(
        R1_boundary,
        R2_boundary,
        linewidth=2.0,
        label="Achievable boundary",
    )
    ax.scatter(
        [R_sym],
        [R_sym],
        zorder=3,
        label=rf"Symmetric rate $R_{{sym}}={R_sym:.4f}$",
    )

    ax.set_xlabel(r"$R_1$  (feedback receiver)")
    ax.set_ylabel(r"$R_2$  (non-feedback receiver)")
    ax.set_title(
        "Single-User Feedback Secure Achievable Region\n"
        rf"$\delta_1={delta1}$, $\delta_2={delta2}$, $\delta_{{1,2}}={delta12:.2f}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax.set_xlim(0, R1_max * 1.08)
    ax.set_ylim(0, R2_max * 1.12)

    text = (
        rf"$R_{{1,\max}}={R1_max:.4f}$" + "\n"
        rf"$R_{{2,\max}}={R2_max:.4f}$" + "\n"
        rf"$R_{{sym}}={R_sym:.4f}$"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename_base}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig. 1] Saved {png_path}")
    print(f"        R1_max={R1_max:.6f}, R2_max={R2_max:.6f}, R_sym={R_sym:.6f}")

    return png_path


# 4. Figure 2: Single feedback vs full feedback vs no feedback

def plot_feedback_comparison(
    delta1=0.4,
    delta2=0.6,
    delta12=None,
    save_dir="figures",
    filename_base="secure_feedback_comparison",
):
    """
    Compare:
      - single-user feedback secure achievable region
      - full-feedback secure capacity region
      - no-feedback secure baseline
    """
    if delta12 is None:
        delta12 = delta1 * delta2

    single_constraints = single_feedback_constraints(delta1, delta2, delta12)
    full_constraints = full_feedback_secure_constraints(delta1, delta2)

    R1_single, R2_single = boundary_from_constraints(single_constraints)
    R1_full, R2_full = boundary_from_constraints(full_constraints)

    baseline = no_feedback_secure_baseline(delta1, delta2)

    single_R1_max, single_R2_max, single_Rsym = region_summary(single_constraints)
    full_R1_max, full_R2_max, full_Rsym = region_summary(full_constraints)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))

    ax.fill_between(
        R1_full,
        0,
        R2_full,
        alpha=0.18,
        label="Full-feedback secure capacity",
    )
    ax.plot(
        R1_full,
        R2_full,
        linewidth=2.0,
    )

    ax.fill_between(
        R1_single,
        0,
        R2_single,
        alpha=0.30,
        label="Single-user feedback secure achievable region",
    )
    ax.plot(
        R1_single,
        R2_single,
        linewidth=2.0,
    )

    if baseline["axis"] == "R1":
        x = np.linspace(0.0, baseline["max_rate"], 300)
        y = np.zeros_like(x)
        ax.plot(x, y, linewidth=3.0, label="No-feedback secure baseline")
        ax.scatter([baseline["max_rate"]], [0.0], zorder=3)
    elif baseline["axis"] == "R2":
        y = np.linspace(0.0, baseline["max_rate"], 300)
        x = np.zeros_like(y)
        ax.plot(x, y, linewidth=3.0, label="No-feedback secure baseline")
        ax.scatter([0.0], [baseline["max_rate"]], zorder=3)
    else:
        ax.scatter([0.0], [0.0], zorder=3, label="No-feedback secure baseline")

    ax.set_xlabel(r"$R_1$")
    ax.set_ylabel(r"$R_2$")
    ax.set_title(
        "Secure Region Comparison:\n"
        "single-user feedback vs full feedback vs no feedback"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    xmax = max(single_R1_max, full_R1_max, baseline["max_rate"]) * 1.12
    ymax = max(single_R2_max, full_R2_max, baseline["max_rate"]) * 1.12
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)

    text = (
        rf"single-user: $R_{{1,\max}}={single_R1_max:.4f},\,R_{{2,\max}}={single_R2_max:.4f}$"
        + "\n"
        + rf"full-feedback: $R_{{1,\max}}={full_R1_max:.4f},\,R_{{2,\max}}={full_R2_max:.4f}$"
        + "\n"
        + rf"no-feedback baseline: {baseline['axis']} max = {baseline['max_rate']:.4f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename_base}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig. 2] Saved {png_path}")

    return png_path


# 5. Figure 3: Secure vs non-secure capacity

def plot_secure_vs_nonsecure_capacity(
    delta1=0.4,
    delta2=0.6,
    delta12=None,
    save_dir="figures",
    filename_base="secure_vs_nonsecure_capacity",
):
    """
    Compare:
      - full-feedback secure capacity region
      - non-secure feedback capacity region
    """
    if delta12 is None:
        delta12 = delta1 * delta2

    secure_constraints = full_feedback_secure_constraints(delta1, delta2)
    nonsecure_constraints = nonsecure_feedback_constraints(delta1, delta2, delta12)

    R1_secure, R2_secure = boundary_from_constraints(secure_constraints)
    R1_nonsecure, R2_nonsecure = boundary_from_constraints(nonsecure_constraints)

    sec_R1_max, sec_R2_max, sec_Rsym = region_summary(secure_constraints)
    nonsec_R1_max, nonsec_R2_max, nonsec_Rsym = region_summary(nonsecure_constraints)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))

    ax.fill_between(
        R1_nonsecure,
        0,
        R2_nonsecure,
        alpha=0.18,
        label="Non-secure feedback capacity",
    )
    ax.plot(
        R1_nonsecure,
        R2_nonsecure,
        linewidth=2.0,
    )

    ax.fill_between(
        R1_secure,
        0,
        R2_secure,
        alpha=0.30,
        label="Full-feedback secure capacity",
    )
    ax.plot(
        R1_secure,
        R2_secure,
        linewidth=2.0,
    )

    ax.set_xlabel(r"$R_1$")
    ax.set_ylabel(r"$R_2$")
    ax.set_title("Secure vs Non-Secure Capacity under Full Feedback")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    xmax = max(sec_R1_max, nonsec_R1_max) * 1.12
    ymax = max(sec_R2_max, nonsec_R2_max) * 1.12
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)

    text = (
        rf"secure: $R_{{1,\max}}={sec_R1_max:.4f},\,R_{{2,\max}}={sec_R2_max:.4f},\,R_{{sym}}={sec_Rsym:.4f}$"
        + "\n"
        + rf"non-secure: $R_{{1,\max}}={nonsec_R1_max:.4f},\,R_{{2,\max}}={nonsec_R2_max:.4f},\,R_{{sym}}={nonsec_Rsym:.4f}$"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename_base}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig. 3] Saved {png_path}")

    return png_path


# 6. Figure 4: Sweep delta2 with delta1 fixed

def sweep_delta2(delta1=0.4, delta2_min=0.05, delta2_max=0.95, num_delta=181):
    """
    Sweep delta2 while using independent erasures:
        delta12 = delta1 * delta2.
    """
    delta2_values = np.linspace(delta2_min, delta2_max, num_delta)

    single_r2max = []
    full_r2max = []
    nofb_r2max = []
    single_r1max = []
    single_rsym = []

    for delta2 in delta2_values:
        delta12 = delta1 * delta2

        single = single_feedback_constraints(delta1, delta2, delta12)
        full = full_feedback_secure_constraints(delta1, delta2)

        single_r2max.append(r2max_from_constraints(single))
        full_r2max.append(r2max_from_constraints(full))
        nofb_r2max.append(no_feedback_r2max(delta1, delta2))
        single_r1max.append(r1max_from_constraints(single))
        single_rsym.append(rsym_from_constraints(single))

    return {
        "delta2": delta2_values,
        "single_r2max": np.array(single_r2max),
        "full_r2max": np.array(full_r2max),
        "nofb_r2max": np.array(nofb_r2max),
        "single_r1max": np.array(single_r1max),
        "single_rsym": np.array(single_rsym),
    }


def plot_delta2_vs_r2max(
    delta1=0.4,
    save_dir="figures",
    filename_base="delta2_vs_R2max",
):
    """
    Plot delta2 versus R2,max, with delta1 fixed.
    """
    data = sweep_delta2(delta1=delta1)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(
        data["delta2"],
        data["single_r2max"],
        linewidth=2.0,
        label="Single-user feedback",
    )
    ax.plot(
        data["delta2"],
        data["full_r2max"],
        linewidth=2.0,
        label="Full feedback",
    )
    ax.plot(
        data["delta2"],
        data["nofb_r2max"],
        linewidth=2.0,
        linestyle="--",
        label="No feedback",
    )
    ax.axvline(delta1, linestyle=":", linewidth=1.8, label=r"$\delta_2=\delta_1$")

    ax.set_xlabel(r"Non-feedback receiver erasure probability $\delta_2$")
    ax.set_ylabel(r"Maximum individual secure rate $R_{2,\max}$")
    ax.set_title(
        rf"Fixed $\delta_1={delta1}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename_base}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


    return png_path


# 7. Figure 5: 3D surface for single-user feedback region

def make_single_feedback_surface(
    delta1=0.4,
    delta2_min=0.05,
    delta2_max=0.95,
    num_delta=121,
    num_r1=180,
):
    """
    Build a 3D surface:
        x-axis: delta2
        y-axis: R1
        z-axis: R2_max(R1, delta2)

    The achievable region for each delta2 is below this surface.
    """
    delta2_values = np.linspace(delta2_min, delta2_max, num_delta)

    r1max_values = []
    for delta2 in delta2_values:
        delta12 = delta1 * delta2
        constraints = single_feedback_constraints(delta1, delta2, delta12)
        r1max_values.append(r1max_from_constraints(constraints))

    global_r1max = max(r1max_values)
    r1_values = np.linspace(0.0, global_r1max, num_r1)

    D2, R1 = np.meshgrid(delta2_values, r1_values, indexing="ij")
    R2 = np.full_like(D2, np.nan, dtype=float)

    for i, delta2 in enumerate(delta2_values):
        delta12 = delta1 * delta2
        constraints = single_feedback_constraints(delta1, delta2, delta12)
        _, r2_boundary = boundary_from_constraints(constraints, R1_grid=r1_values)
        R2[i, :] = r2_boundary

    return D2, R1, R2



def plot_single_feedback_3d_surface(
    delta1=0.4,
    save_dir="figures",
    filename_base="single_feedback_region_surface_3d",
    interactive=True,
):
    """
    Plot 3D surface showing how the single-user feedback achievable boundary
    changes with delta2.

    If interactive=True, the figure window will stay open
    """
    D2, R1, R2 = make_single_feedback_surface(delta1=delta1)

    fig = plt.figure(figsize=(8.0, 5.9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(D2, R1, R2, linewidth=0, antialiased=True, alpha=0.9)

    ax.set_xlabel(r"$\delta_2$")
    ax.set_ylabel(r"$R_1$")
    ax.set_zlabel(r"$R_{2,\max}(R_1,\delta_2)$")
    ax.set_title(
        rf"Single-user feedback achievable boundary, fixed $\delta_1={delta1}$"
    )
    ax.view_init(elev=22, azim=65)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename_base}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"[Fig. 5] Saved {png_path}")
    if interactive:
        print("Interactive 3D window opened. Drag with the mouse to rotate the figure.")
        plt.show()
    else:
        plt.close(fig)

    return png_path


# 8. sweep delta1 with delta2 fixed

def sweep_delta1_paper_style(
    delta2=0.4,
    delta1_min=0.05,
    delta1_max=0.95,
    num_delta=181,
):
    """
    Paper-style sweep:
        fix delta2, sweep delta1,
        delta12 = delta1 * delta2

    studies how the feedback receiver channel affects
    the non-feedback receiver's individual secure rate
    """
    delta1_values = np.linspace(delta1_min, delta1_max, num_delta)

    single_r2max = []
    full_r2max = []
    nofb_r2max = []

    for delta1 in delta1_values:
        delta12 = delta1 * delta2
        single = single_feedback_constraints(delta1, delta2, delta12)
        full = full_feedback_secure_constraints(delta1, delta2)

        single_r2max.append(r2max_from_constraints(single))
        full_r2max.append(r2max_from_constraints(full))
        nofb_r2max.append(no_feedback_r2max(delta1, delta2))

    return {
        "delta1": delta1_values,
        "single_r2max": np.array(single_r2max),
        "full_r2max": np.array(full_r2max),
        "nofb_r2max": np.array(nofb_r2max),
    }


def plot_delta1_vs_r2max_paper_style(
    delta2=0.4,
    save_dir="figures",
    filename_base="delta1_vs_R2max_paper_style",
):
    """
    Plot paper-style delta1 sweep with fixed delta2
    """
    data = sweep_delta1_paper_style(delta2=delta2)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(
        data["delta1"],
        data["single_r2max"],
        linewidth=2.0,
        label="Single-user feedback",
    )
    ax.plot(
        data["delta1"],
        data["full_r2max"],
        linewidth=2.0,
        label="Full feedback",
    )
    ax.plot(
        data["delta1"],
        data["nofb_r2max"],
        linewidth=2.0,
        linestyle="--",
        label="No feedback",
    )
    ax.axvline(delta2, linestyle=":", linewidth=1.8, label=r"$\delta_1=\delta_2$")

    ax.set_xlabel(r"Feedback receiver erasure probability $\delta_1$")
    ax.set_ylabel(r"Maximum individual secure rate $R_{2,\max}$")
    ax.set_title(
        rf"Fixed non-feedback receiver $\delta_2={delta2}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename_base}.png"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


    return png_path


# 9. Main driver

def generate_all_figures(
    delta1=0.4,
    delta2=0.6,
    delta12=None,
    save_dir="figures",
):
    """
    Generate all figures used in the numerical section
    """
    if delta12 is None:
        delta12 = delta1 * delta2

    outputs = {}

    outputs["single_feedback_region"] = plot_single_feedback_region(
        delta1=delta1,
        delta2=delta2,
        delta12=delta12,
        save_dir=save_dir,
    )

    outputs["feedback_comparison"] = plot_feedback_comparison(
        delta1=delta1,
        delta2=delta2,
        delta12=delta12,
        save_dir=save_dir,
    )

    outputs["secure_vs_nonsecure"] = plot_secure_vs_nonsecure_capacity(
        delta1=delta1,
        delta2=delta2,
        delta12=delta12,
        save_dir=save_dir,
    )

    outputs["delta2_sweep"] = plot_delta2_vs_r2max(
        delta1=delta1,
        save_dir=save_dir,
    )

    outputs["surface_3d"] = plot_single_feedback_3d_surface(
        delta1=delta1,
        save_dir=save_dir,
        interactive=True,
    )

    outputs["delta1_paper_style_sweep"] = plot_delta1_vs_r2max_paper_style(
        delta2=0.4,
        save_dir=save_dir,
    )

    return outputs


if __name__ == "__main__":
    # delta12 = delta1 * delta2 corresponds to independent erasures.
    generate_all_figures(
        delta1=0.4,
        delta2=0.6,
        delta12=0.24,
        save_dir="figures",
    )