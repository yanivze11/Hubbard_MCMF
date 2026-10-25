# fh_run_n.jl — what one SLURM job runs: one simulation at fixed filling.
#
#   julia fh_run_n.jl --sim_path=<campaign>/<i>
#
# Reads <sim_path>/sim.jld2 (written by an fh_pre_run_*.jl driver) and calls
# running_simulation_wanted_occupation with it (occupation_convergence.jl): μ search,
# then the production run in <sim_path>/seed_<s>/. If a checkpoint exists the run
# resumes from it, so a requeued or resubmitted job continues the same chain.
# Parameters missing from older sim.jld2 files get defaults that reproduce the old runs.

using Statistics, Dates, Base.Iterators, JLD2, ArgParse

include("occupation_convergence.jl")

"""
    run_me(sim_path)

Load `sim_path/sim.jld2` and run (or resume) its simulation for its first MC seed with
`running_simulation_wanted_occupation`, saving into `sim_path/seed_<seed>/`.
"""
function run_me(sim_path)
    sim_data_file=joinpath(sim_path,"sim.jld2")
    sim_data_dict = JLD2.load(sim_data_file)

    running_simulation_wanted_occupation(
                            sim_data_dict["Lx"],
                            sim_data_dict["Ly"],
                            sim_data_dict["SC_L"],
                            sim_data_dict["U"],
                            sim_data_dict["t"],
                            sim_data_dict["V"],
                            sim_data_dict["seeds"][1],  # Single seed per job
                            sim_data_dict["wanted_n"],
                            sim_data_dict["beta"],
                            sim_data_dict["B"],
                            sim_data_dict["num_measure"],
                            sim_data_dict["num_thermal"],
                            sim_data_dict["measuring_interval"],
                            sim_data_dict["w_list"],
                            sim_data_dict["guess"],
                            sim_data_dict["U_disorder"],
                            sim_data_dict["diag_method"],
                            sim_data_dict["pairing_symmetry"],
                            sim_data_dict["w_conductivity_range"],
                            sim_data_dict["impurity"];

                            # Which disorder realization to draw. Absent in
                            # pre-2026-07 sim.jld2 files, where 0 reproduces the
                            # old behaviour of deriving it from the MC seed.
                            disorder_seed=Int64(get(sim_data_dict, "disorder_seed", 0)),

                            directory=sim_path,
                            resume_path=sim_path,
                            num_cores=sim_data_dict["cores"],
                            savedata=true,
                            saving_interval=500,
                            saveplots=false,
                            date_folder=false,
                            seed_folder=true,
                            dump_time=Dates.Hour(1),
                            numbered=false,
                            U_puddle_params=sim_data_dict["U_puddle_params"],
                            TClustersize_x=sim_data_dict["TClustersize"],
                            TClustersize_y=sim_data_dict["TClustersize"],
                            U_smooth=sim_data_dict["smooth_U"],
                            # Occupation convergence parameters
                            num_thermal_occ=sim_data_dict["num_thermal_occ"],
                            num_measure_occ=sim_data_dict["num_measure_occ"],
                            measuring_interval_occ=sim_data_dict["measuring_interval_occ"],
                            n_tol=sim_data_dict["n_tol"],
                            max_iter=sim_data_dict["max_iter"],
                            num_uncorrelated_measurments=sim_data_dict["num_uncorrelated_measurments"],
                            boundary_conditions=sim_data_dict["boundary_condition"],
                            gauge_choice=sim_data_dict["gauge_choice"],
                            delta_fixed_values=sim_data_dict["delta_fixed_values"],
                            delta_values_stiffness=sim_data_dict["delta_values_stiffness"],
                            dw_values=sim_data_dict["dw_values"],

                            # Density-channel (phi) decoupling. Absent in pre-2026-07
                            # sim.jld2 files, where the defaults reproduce the legacy
                            # (channel-off) run exactly.
                            alpha=Float64(get(sim_data_dict, "alpha", 1.0)),
                            beta_phi=Float64(get(sim_data_dict, "beta_phi", 0.0)),
                            Q_density=Float64[get(sim_data_dict, "Qx", 0.0),
                                              get(sim_data_dict, "Qy", 0.0)],

                            # Coupled / potential mu-U disorder + field observables.
                            # Absent in older sim.jld2 files, where the defaults leave
                            # every feature off (identical to the legacy run).
                            mu_disorder=String(get(sim_data_dict, "mu_disorder", "none")),
                            mu_disorder_f=Float64(get(sim_data_dict, "mu_disorder_f", 0.0)),
                            snapshot_interval=Int64(get(sim_data_dict, "snapshot_interval", 0)),
                            phi_hist_range=Float64.(get(sim_data_dict, "phi_hist_range", [-10.0, 10.0])),
                            phi_hist_nbins=Int64(get(sim_data_dict, "phi_hist_nbins", 0)),
                            Delta_hist_abs_range=Float64.(get(sim_data_dict, "Delta_hist_abs_range", [0.0, 10.0])),
                            Delta_hist_abs_nbins=Int64(get(sim_data_dict, "Delta_hist_abs_nbins", 0)),
                            Delta_hist_arg_nbins=Int64(get(sim_data_dict, "Delta_hist_arg_nbins", 0))
                            )
end

s = ArgParseSettings()
@add_arg_table s begin
     "--sim_path"
         help = "sim_path"
end

parsed_args = parse_args(ARGS, s)
sim_path = parsed_args["sim_path"]

run_me(sim_path)
