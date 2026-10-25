# Demo: MC-field histograms on a small system.
#
# Runs a small s-wave lattice with the density channel ON, then plots the lifetime
# field histograms straight out of the checkpoint:
#   * phi_i               -> 1D histogram
#   * (|Delta_s|, arg)    -> 2D JOINT histogram (amplitude x phase), plus its marginals
#
# Run from anywhere:  julia --project=. demos/field_histograms_demo.jl
# Output PNGs land in demos/output/ (gitignored).
ENV["GKSwstype"] = "100"  # headless GR (savefig only, no window)

include(joinpath(@__DIR__, "..", "MC_Hubburd.jl"))

using Plots
using Printf
gr()

const OUT = joinpath(@__DIR__, "output")
mkpath(OUT)

# --- run a small system ------------------------------------------------------
# L=6 s-wave, density channel ON (alpha=1, beta_phi=0.5, uniform Hartree Q=0).
# Observables are trimmed to the minimum (histograms don't need DOS/conductivity),
# so the run is fast; histograms use the default binning (phi 100, Delta 100x50).
run_dir = joinpath(OUT, "hist_demo_run")
isdir(run_dir) && rm(run_dir; recursive=true)
mkpath(run_dir)

sim = SimData(
    Lx=6, Ly=6, SC_L=1, U=5.0, t=1.0, V=0.0, seed=12345, mu=0.0, beta=10.0,
    num_measure=1200, num_thermal=300, measuring_interval=1, saving_interval=200,
    w_list=[0.0], guess="uniform", U_disorder="uniform",
    diag_method="K0", pairing_symmetry="s-wave", directory=run_dir,
    num_cores=1, num_threads=1, w_conductivity_range=[0.0],
    delta_values_fixed=Float64[], delta_values_stiffness=Float64[], dw_values=Float64[],
    U_disorder_params=[0.0, 10.0, 0.7, 0.7], U_impurity=0.0,
    TCluster_size_x=1, TCluster_size_y=1, smooth_U=(false, 0.8),
    loop_random=true, B=0.0, gauge_choice="LandauX", boundary_conditions="PBC",
    compute_jP=false, alpha=1.0, beta_phi=0.5, Q_density=[0.0, 0.0],
    snapshot_interval=0,   # snapshots off for this demo
    # histograms: defaults (phi 100 bins [-10,10]; Delta 100x50 over [0,10]x[0,2pi])
)

println("Running L=$(sim.Lx) s-wave, beta=$(sim.beta), U=$(sim.U), density channel on ...")
t0 = time()
data = MonteCarloSimulation(
    sim, run_dir; saveplots=false, savedata=true, saving_interval=sim.saving_interval,
    dump_time=Dates.Hour(1), sim_info_path=joinpath(run_dir, "sim.jld2"),
    log_filename=joinpath(run_dir, "log.txt"),
)
@printf("done in %.1f s, measurements = %d\n", time() - t0, data.current_state.num_current_measurments)

# --- pull histograms back from the checkpoint (source of truth) --------------
loaded = load_checkpoint(run_dir)
cs = loaded.current_state
sd = loaded.sim_data
phi_hist   = cs.phi_hist
Delta_hist = cs.Delta_hist        # (amplitude bins) x (phase bins)
M = cs.num_current_measurments
N = sd.N

"""
    bin_centers(lo, hi, n)

Centers of `n` equal bins over [lo, hi] (the layout of the field histograms).
"""
bin_centers(lo, hi, n) = (edges = range(lo, hi; length=n + 1); (edges[1:end-1] .+ edges[2:end]) ./ 2)
phi_c   = bin_centers(sd.phi_hist_range[1], sd.phi_hist_range[2], sd.phi_hist_nbins)
amp_c   = bin_centers(sd.Delta_hist_abs_range[1], sd.Delta_hist_abs_range[2], sd.Delta_hist_abs_nbins)
phase_c = bin_centers(0.0, 2pi, sd.Delta_hist_arg_nbins)

# marginals (for the side panels + amplitude cropping)
amp_marg   = vec(sum(Delta_hist; dims=2))   # length = amp bins
phase_marg = vec(sum(Delta_hist; dims=1))   # length = phase bins

# crop the amplitude axis to the populated region so the heatmap is readable
last_amp = findlast(>(0), amp_marg)
last_amp = last_amp === nothing ? length(amp_c) : min(last_amp + 3, length(amp_c))
amp_idx = 1:last_amp

@printf("click totals: phi=%d, Delta=%d (expected %d each)\n", sum(phi_hist), sum(Delta_hist), M * N)

# --- plots -------------------------------------------------------------------
p_phi = bar(phi_c, phi_hist; legend=false, color=:steelblue, linecolor=:steelblue,
            xlabel="phi_i", ylabel="counts", title="phi histogram (1D)")
nz_phi = findall(>(0), phi_hist)
if !isempty(nz_phi)
    xlims!(p_phi, phi_c[max(1, first(nz_phi) - 3)], phi_c[min(end, last(nz_phi) + 3)])
end

p_amp = bar(amp_c[amp_idx], amp_marg[amp_idx]; legend=false, color=:seagreen, linecolor=:seagreen,
            xlabel="|Delta_s|", ylabel="counts", title="amplitude marginal")

p_phase = bar(phase_c, phase_marg; legend=false, color=:indianred, linecolor=:indianred,
              xlabel="arg(Delta_s)", ylabel="counts", title="phase marginal",
              xticks=([0, pi/2, pi, 3pi/2, 2pi], ["0", "pi/2", "pi", "3pi/2", "2pi"]))

p_2d = heatmap(phase_c, amp_c[amp_idx], Delta_hist[amp_idx, :];
               xlabel="arg(Delta_s)", ylabel="|Delta_s|",
               title="Delta_s 2D joint histogram", color=:viridis, colorbar_title="counts",
               xticks=([0, pi/2, pi, 3pi/2, 2pi], ["0", "pi/2", "pi", "3pi/2", "2pi"]))

fig = plot(p_phi, p_amp, p_phase, p_2d; layout=(2, 2), size=(1100, 850),
           plot_title=@sprintf("L=%dx%d  s-wave  U=%.1f  beta=%.1f  |  %d measurements x %d sites",
                                 sd.Lx, sd.Ly, sd.U, sd.beta, M, N))

out_png = joinpath(OUT, "field_histograms_demo.png")
savefig(fig, out_png)
savefig(p_2d, joinpath(OUT, "delta_joint_heatmap.png"))
println("saved: ", out_png)
