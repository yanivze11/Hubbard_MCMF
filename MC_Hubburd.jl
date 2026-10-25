# MC_Hubburd.jl — the simulation engine: Monte-Carlo mean field (MCMF) for the 2D
# attractive Hubbard model with a static pairing field Δ_i (and optional density
# field φ_i). The model, the action and the diag_method options are explained in
# README.md.
#
# Sections (each a `begin # ...` block):
#   indexing and nearest neighbors   lattice indices, hopping matrices, k lists
#   sim structure                    SimData (all run parameters) and the run-time structs
#   structs and measurments          MeasureData, initial state, LatticeData, U/V disorder,
#                                    BdG Hamiltonian + action (MCMF_action_*), measurements
#   Saving helper functions          checkpoints, results.jld2 chunks, load_results
#   update suggestion / saddle point proposal of a new field value; mean-field saddle point
#   spectral function                A(k, ω), DOS, LDOS
#   MC simulation                    MC sweep (MC_stepV2!), thermalization, measurement loop
#   plotting functions               quick-look plots of one run
#   file name / running ...          output folders, running_simulation_once/_seeds
#   computing superfluid stiffness   current-current correlations Λ(q, ω), conductivity
#   U_puddles                        quick looks at the U-disorder maps
# Fixed-filling runs (μ search) are in occupation_convergence.jl; means and error bars
# in sim_num_control.jl.

include("statistics.jl")
include("dwave-kspaceMF.jl")  # D-wave mean field calculations
include("mf_kspace_density_channel.jl")  # k-space mean field for coupled Delta + phi (Q=0)
using .MFKSpaceDensityChannel

begin # using
	using LinearAlgebra
	using MKL
	using Plots
    using SparseArrays
    using JLD2
    using Dates
	using Distributions
	using Random
	using LinearAlgebra.LAPACK
	using FastLapackInterface
	using Measures
	using FFTW
	using ImageFiltering
	using JSON
end


begin # indexing and nearest neighbors
	
	# Sites are numbered i = (y - 1)·Lx + x, with x ∈ 1:Lx, y ∈ 1:Ly.

	"""
	    add_xy_ix(coord, d_coord, L)

	Move a coordinate by `d_coord` with periodic wrapping into 1:L.
	"""
	add_xy_ix(coord, d_coord, L) = mod1(coord + d_coord, L)

	"""
	    make_CartesianIndices(Lx, Ly)

	Converter site number -> (x, y) for an Lx×Ly lattice (stored in `ConstStruct`).
	"""
	make_CartesianIndices(Lx::Int64, Ly::Int64) = CartesianIndices((Lx, Ly))

	"""
	    make_LinearIndices(Lx, Ly)

	Converter (x, y) -> site number for an Lx×Ly lattice (stored in `ConstStruct`).
	"""
	make_LinearIndices(Lx::Int64, Ly::Int64) = LinearIndices((Lx, Ly))

	"""
	    get_xy(ix, CI) -> (x, y)

	Coordinates of site `ix`.
	"""
	get_xy(ix::Int64, CI::CartesianIndices) = Tuple(CI[ix])

	"""
	    get_x(ix, CI)
	    get_x(ix, Lx)

	x coordinate of site `ix` (from the index converter, or from Lx alone).
	"""
	get_x(ix::Int64, CI::CartesianIndices) = CI[ix][1]

	"""
	    get_y(ix, CI)
	    get_y(ix, Lx)

	y coordinate of site `ix` (from the index converter, or from Lx alone).
	"""
	get_y(ix::Int64, CI::CartesianIndices) = CI[ix][2]

	"""
	    get_ix(x, y, LI)

	Site number of (x, y).
	"""
	get_ix(coord_x::Int64, coord_y::Int64, LI::LinearIndices) = LI[coord_x, coord_y]

	"""
	    idx(x, y, Lx)

	Site number of (x, y), from Lx alone: (y - 1)·Lx + x.
	"""
	idx(coord_x::Int64, coord_y::Int64, Lx::Int64) = (coord_y - 1) * Lx + coord_x

	get_x(ix::Int64, Lx::Int64) = mod1(ix, Lx)

	get_y(ix::Int64, Lx::Int64) = div(ix - 1, Lx) + 1

	"""
	    neighbor_matrix(Lx, Ly) -> SparseMatrixCSC
	    neighbor_matrix(L)

	Nearest-neighbor adjacency matrix of the periodic Lx×Ly lattice (1 for each of the
	4 neighbors), no phases.
	"""
	function neighbor_matrix(Lx::Int64, Ly::Int64)
	    N = Lx * Ly
	    A = spzeros(N, N)  # Sparse matrix initialization
		Neigh = [[1,0], [0,1], [-1,0], [0,-1]]

		for ix in 1:N
			my_x = get_x(ix, Lx)
			my_y = get_y(ix, Lx)

			for n in Neigh
				new_x = add_xy_ix(my_x, n[1], Lx)
				new_y = add_xy_ix(my_y, n[2], Ly)
				
				new_ix = idx(new_x, new_y, Lx)
				A[ix, new_ix] = 1
			end
		end
	    
	    return A
	end
	
	# Legacy single-L version for backward compatibility
	neighbor_matrix(L::Int64) = neighbor_matrix(L, L)

	"""
	    neighbor_matrix(Lx, Ly, K; B=0.0, gauge_choice="LandauX") -> SparseMatrixCSC{ComplexF64}
	    neighbor_matrix(L, K; ...)

	Hopping matrix of the periodic Lx×Ly lattice with phases.

	- Twisted boundary: a hop that crosses the x (y) boundary gets e^{±i K_x L_x}
	  (e^{±i K_y L_y}); `K` is the supercell wavevector (`[0, 0]` for `SC_L = 1`).
	- Magnetic field `B` (Peierls phases), by `gauge_choice`:
	  `"LandauX"`: e^{-iBy} on +x hops; `"LandauY"`: e^{iBx} on +y hops;
	  `"Mixed"`: e^{iB·y·Lx} on +x hops and e^{iB·x·Ly} on +y hops.
	  Reverse hops get the complex conjugate.
	"""
	function neighbor_matrix(Lx::Int64, Ly::Int64, K::Array{Float64,1}; B::Float64=0.0, gauge_choice::String="LandauX")
	    N = Lx * Ly
	    A = spzeros(ComplexF64, N, N)  # Sparse matrix initialization
		Neigh = [[1,0], [0,1], [-1,0], [0,-1]]
		# Phase sites: where boundary crossing occurs
		phase_site = [[Lx,0], [0,Ly], [1,0], [0,1]]
		# Phase factors for crossing boundaries (x-direction uses Lx, y-direction uses Ly)
		phase_factor = [
			exp(1im * K[1] * Lx),   # +x direction crossing
			exp(1im * K[2] * Ly),   # +y direction crossing
			exp(-1im * K[1] * Lx),  # -x direction crossing
			exp(-1im * K[2] * Ly)   # -y direction crossing
		]
	    
		for ix in 1:N
			my_x = get_x(ix, Lx)
			my_y = get_y(ix, Lx)

			for (n_ix, n) in enumerate(Neigh)
				new_x = add_xy_ix(my_x, n[1], Lx)
				new_y = add_xy_ix(my_y, n[2], Ly)
				
				new_ix = idx(new_x, new_y, Lx)
				
				if my_x == phase_site[n_ix][1] || my_y == phase_site[n_ix][2]
					A[ix, new_ix] = phase_factor[n_ix]
				else
					A[ix, new_ix] = 1.0
				end

				if B != 0.0
					# Peierls phase factor
					if gauge_choice == "LandauX"
						phase_x = exp(-1im * B * my_y)
						if n == [1, 0]
							A[ix, new_ix] *= phase_x
						elseif n == [-1, 0]
							A[ix, new_ix] *= conj(phase_x)
						end
					elseif gauge_choice == "LandauY"
						# phase for hopping in y direction
						phase_y = exp(1im * B * my_x)
						if n == [0, 1]
							A[ix, new_ix] *= phase_y
						elseif n == [0, -1]
							A[ix, new_ix] *= conj(phase_y) # changed to A[new_ix, ix] from A[ ix, new_ix]
						end
					elseif gauge_choice == "Mixed"
						# gauge that has both x and y components
						# with Ly=Lx+1 we get high resolution magnetic field values
						# vector potential A = 2*pi*q*(y/Ly ,x/Lx) = B*(y*Lx, x*Ly)
						# Magnetic field:B = 2*pi*q/(Lx*Ly)
						# phase for hopping in x direction
						phase_x = exp(1im * B * my_y*Lx)
						# phase for hopping in y direction
						phase_y = exp(1im * B * my_x*Ly)
						if n == [1, 0]
							A[ix, new_ix] *= phase_x
						elseif n == [-1, 0]
							A[ix, new_ix] *= conj(phase_x) # changed to A[new_ix, ix] from A[ ix, new_ix]
						elseif n == [0, 1]
							A[ix, new_ix] *= phase_y
						elseif n == [0, -1]
							A[ix, new_ix] *= conj(phase_y) # changed to A[new_ix, ix] from A[ ix, new_ix]
						end
					else 
						error("gauge_choice must be 'LandauX' , 'LandauY' or 'Mixed'")
					end
				
					
				end
			end
		end
	    
	    return A
	end
	
	# Legacy single-L version for backward compatibility
	neighbor_matrix(L::Int64, K::Array{Float64,1}; B::Float64=0.0, gauge_choice::String="LandauX") = neighbor_matrix(L, L, K; B=B, gauge_choice=gauge_choice)

	"""
	    neighbor_matrix_x(Lx, Ly, direction) -> Matrix{ComplexF64}
	    neighbor_matrix_x(L, direction)

	Dense matrix of the +x (`direction = "x"`) or +y (`"y"`) hops only, no phases
	(used for the bond pairing fields of d-wave).
	"""
	function neighbor_matrix_x(Lx::Int64, Ly::Int64, direction::String)
	    N = Lx * Ly
	    A = zeros(ComplexF64, N, N)  # Dense matrix initialization
		n = direction == "x" ? [1,0] : [0,1]

		for ix in 1:N
			my_x = get_x(ix, Lx)
			my_y = get_y(ix, Lx)

			new_x = add_xy_ix(my_x, n[1], Lx)
			new_y = add_xy_ix(my_y, n[2], Ly)
		
			new_ix = idx(new_x, new_y, Lx)
			A[ix, new_ix] = 1.0
		end
	    
	    return A	
	end
	
	# Legacy single-L version for backward compatibility
	neighbor_matrix_x(L::Int64, direction::String) = neighbor_matrix_x(L, L, direction)

	"""
	    neighbor_matrix_K_vector(Lx, Ly, K_vectors; B=0.0, gauge_choice="LandauX")

	`neighbor_matrix(Lx, Ly, K; ...)` for each supercell wavevector K in `K_vectors`
	(a single K = 0 for `SC_L = 1`).
	"""
	function neighbor_matrix_K_vector(Lx::Int64, Ly::Int64, K_vectors::Array{Array{Float64,1},1}; B::Float64=0.0, gauge_choice::String="LandauX")
		NN_matrices = [neighbor_matrix(Lx, Ly, k_vec; B=B, gauge_choice=gauge_choice) for k_vec in K_vectors]
		return NN_matrices
	end
	
	# Legacy single-L version for backward compatibility
	neighbor_matrix_K_vector(L::Int64, K_vectors::Array{Array{Float64,1},1}; B::Float64=0.0, gauge_choice::String="LandauX") = neighbor_matrix_K_vector(L, L, K_vectors; B=B, gauge_choice=gauge_choice)

	"""
	    get_kx_list(Lx)

	Allowed momenta 2πm/Lx, m = -⌊Lx/2⌋ … Lx-⌊Lx/2⌋-1 (k = 0 at index ⌊Lx/2⌋+1, k = -π at index 1).
	"""
	get_kx_list(Lx::Int64) = 2*pi/Lx * (-div(Lx,2):(Lx - div(Lx,2) - 1))

	"""
	    get_ky_list(Ly)

	Same as `get_kx_list`, for the y direction.
	"""
	get_ky_list(Ly::Int64) = 2*pi/Ly * (-div(Ly,2):(Ly - div(Ly,2) - 1))

	"""
	    q_sig_component(v) -> Int64

	`v` rounded to 1e-12, so that equal q values compare equal despite floating-point noise.
	"""
	@inline q_sig_component(v::Float64) = round(Int64, v * 1e12)

	"""
	    q_list_signature(q_vecs) -> UInt64

	Hash of a list of q vectors (FNV-1a over the rounded components), used to check
	whether a cached Fourier table belongs to the same q list.
	"""
	function q_list_signature(q_vecs::Array{Array{Float64,1},1})
		h = UInt64(0xcbf29ce484222325)
		for q in q_vecs
			h = (h ⊻ UInt64(q_sig_component(Float64(q[1])))) * UInt64(0x100000001b3)
			h = (h ⊻ UInt64(q_sig_component(Float64(q[2])))) * UInt64(0x100000001b3)
		end
		return (h ⊻ UInt64(length(q_vecs)))
	end

	"""
	    fill_FT_table!(FT, q_vecs, X, Y)

	Fill the N × n_q table FT[i, q] = e^{i (q_x X_i + q_y Y_i)}.
	"""
	function fill_FT_table!(FT::Array{ComplexF64,2}, q_vecs::Array{Array{Float64,1},1}, X::Array{Int64,1}, Y::Array{Int64,1})
		N = length(X)
		q_length = length(q_vecs)
		@inbounds for iq in 1:q_length
			qx = q_vecs[iq][1]
			qy = q_vecs[iq][2]
			@simd for ix in 1:N
				FT[ix, iq] = exp(1im * (qx * X[ix] + qy * Y[ix]))
			end
		end
		return FT
	end
	
	"""
	    get_ki_list(L)

	Old name of `get_kx_list`.
	"""
	get_ki_list(L) = get_kx_list(L)
end


begin # sim structure
	"""
	    SimData

	All parameters of one run (one MC seed). Stored in sim.jld2 and in every
	checkpoint; docs/WORKFLOW.md Appendix A explains each parameter.
	Build it with the keyword constructor `SimData(; Lx, ...)` for tests, or let
	`running_simulation_once` / `running_simulation_wanted_occupation` build it.
	"""
	mutable struct SimData
		# lattice size in x direction
	    Lx::Int64
		# lattice size in y direction
		Ly::Int64
		# total number of sites (Lx * Ly)
		N::Int64
		# supercell size (abandoned supercell method; keep 1)
		SC_L::Int64
	    # coupling strength
	    U::Float64
		# hopping term
		t::Float64
		# V -  disorder strength
		V::Float64
		# Seed for random number generator
		seed::Int64
		# chemical potential
		mu::Float64
		#inverse temperature
		beta::Float64
	    # numbers of measurements
	    num_measure::Int64
	    # numbers of themalization steps
	    num_thermal::Int64
		# measurment interval
		measuring_interval::Int64
		# saving_interval
		saving_interval::Int64
		#list of energies for DOS measurment
		w_list::Array{Float64,1}
        #initial guess type for delta -> can be "rand" or "uniform"
        guess::String
		# U disorder model -> see make_U_params
		U_disorder::String
		# diagonalization method -> "K0" or "K0-TCA" ("exact"/"TCluster": abandoned supercell methods)
		diag_method::String
		# Pairing symmetry -> "s-wave" (validated); "d-wave", "d-p-wave", "mayr-d-wave", "d+s-wave" (experimental)
		pairing_symmetry::String
		# directory for saving data
		directory::String
		#number of cores to use
		num_cores::Int64
		# num_threads for BLAS
		num_threads::Int64
		#start time of the simulation
		start_time::DateTime
		# energy range for conductivity calculations
		w_conductivity_range::Array{Float64,1}
		# Fixed delta values for conductivity measurements (constant broadening parameters)
		delta_values_fixed::Array{Float64,1}
		# Fixed delta values for stiffness measurements: REMOVED feature, kept so old files load
		delta_values_stiffness::Array{Float64,1}
		# U disorder parameters : (V0, VI, lambda_U, rz), see make_U_params
		U_disorder_params::Array{Float64,1}
		# U_impurity : disorder strength; its meaning depends on the U_disorder model (see make_U_params)
		U_impurity::Float64
		# traveling cluster size in x direction (K0-TCA)
		TCluster_size_x::Int64
		# traveling cluster size in y direction
		TCluster_size_y::Int64
		# smooth U parameters - (true/false , smooth_sigma)
		smooth_U::Tuple{Bool, Float64}
		# loop random - true/false for random site and pairing updates
		loop_random::Bool
		# magnetic field in Z direction, in units of flux per plaquette (quantized, see assert_flux_quantization!)
		B::Float64
		# gauge for the magnetic field: "LandauX" (phases on x bonds), "LandauY" (on y bonds), "Mixed" (both); see neighbor_matrix
		gauge_choice::String
		# Boundary conditions - "PBC" or "Y-PBC"
		boundary_conditions::String # for now only "PBC" is implemented
		# Fixed Δω values for DC conductivity (delta->0 integrated formula)
		dw_values::Array{Float64,1}
		# Whether to compute the FT paramagnetic current observable (jP). Default off.
		compute_jP::Bool
		# Which DISORDER REALIZATION to draw (potential disorder V_i and the U
		# disorder map), independent of the Monte-Carlo seed. 0 = legacy
		# behaviour: derive it from `seed`, so old runs reproduce exactly.
		#
		# Keeping this separate from `seed` is what lets you average Monte-Carlo
		# noise at FIXED disorder, and vary the disorder sample independently —
		# which is the whole point when the sample-to-sample spread IS the signal.
		disorder_seed::Int64
		# --- density-channel (Hubbard-Stratonovich) decoupling --------------------
		# Fraction of U kept in the PAIRING channel. The |Delta|^2 penalty becomes
		# |Delta_i|^2 / (alpha * U_i). alpha = 1 reproduces the legacy behaviour
		# (all of U decoupled in the pairing channel, penalty |Delta|^2/U).
		alpha::Float64
		# Fraction of U kept in the DENSITY channel. The phi field carries a penalty
		# phi_i^2 / (2 * beta_phi * U_i). A strict partition has alpha + beta_phi = 1;
		# the equal split (equal penalty denominators) is alpha=2/3, beta_phi=1/3.
		# beta_phi = 0 turns the density channel OFF (no phi field, legacy behaviour).
		beta_phi::Float64
		# Ordering vector Q of the density coupling: phi couples as
		# cos(Q.r_i) phi_i * n_i, realised as a local chemical-potential shift
		# mu -> mu + cos(Q.r_i) phi_i. [0,0] = uniform (Hartree), [pi,pi] = staggered
		# charge-density-wave (checkerboard +-1).
		Q_density::Array{Float64,1}
		# How often (measured in COMPLETED measurements) to dump a raw snapshot of
		# the Monte-Carlo fields (Delta_s/dx/dy, phi) to a SEPARATE
		# field_snapshots.jld2. 0 => OFF (legacy checkpoints load as 0). Independent
		# of saving_interval / measuring_interval.
		snapshot_interval::Int64
		# --- MC-field histograms (lifetime accumulators, see CurrentState.*_hist) ---
		# Every measurement each site is binned. Values outside a range are CLAMPED
		# into the edge bin (so the first/last bin also count everything below/above
		# the range). nbins == 0 => that histogram is OFF (legacy checkpoints load all
		# nbins as 0). Delta uses the s-wave field Delta_s only.
		# phi: 1D histogram, [lo, hi] range and bin count.
		phi_hist_range::Array{Float64,1}
		phi_hist_nbins::Int64
		# Delta_s: 2D JOINT histogram of (|Delta_s|, arg(Delta_s)) -- amplitude and
		# phase binned TOGETHER. Amplitude axis: [lo, hi] range + bin count. Phase
		# axis: bin count over the fixed range [0, 2pi]. The joint histogram is ON
		# only when BOTH bin counts are > 0.
		Delta_hist_abs_range::Array{Float64,1}
		Delta_hist_abs_nbins::Int64
		Delta_hist_arg_nbins::Int64
		# --- coupled / potential mu-U disorder -----------------------------------
		# Binary mu-U disorder with U_i in {0, U}. A fraction f of sites (EXACT count
		# floor(f*N)) sits at U_i = 0; the remaining (1-f) at U_i = U. Modes:
		#   "none"             : off (legacy). mu_shift = 0 everywhere.
		#   "interaction-only" : U_params set to {0,U} on the partition; mu NOT shifted.
		#   "coupled"          : U_params set to {0,U} AND mu_i = mu - U_i/2. (s-wave.)
		#   "potential-only"   : interaction stays uniform U; only mu is disordered
		#                        (mu -> mu - U/2 on the (1-f) sites).
		# The trio {interaction-only, coupled, potential-only} share one site partition
		# at fixed (f, disorder_seed), isolating interaction- vs potential-disorder.
		# The partition is drawn from disorder_seed (independent of the MC seed).
		mu_disorder::String
		# Fraction f of sites with U_i = 0 (mu unshifted). Only used when
		# mu_disorder != "none". f=0 => all sites shifted; f=1 => none shifted.
		mu_disorder_f::Float64
	end

	"""
	    SimData(; Lx, Ly=Lx, kwargs...)

	Keyword constructor with defaults for every parameter (for tests and interactive use).
	`N = Lx*Ly` is filled in. Note that some defaults differ from the drivers' defaults
	(e.g. snapshots and histograms are on here).
	"""
	function SimData(;
		Lx::Int64,
		Ly::Int64 = Lx,  # Default to square if Ly not specified
		SC_L::Int64 = 1,
		U::Float64 = 4.0,
		t::Float64 = 1.0,
		V::Float64 = 0.0,
		seed::Int64 = 12345,
		mu::Float64 = 0.0,
		beta::Float64 = 10.0,
		num_measure::Int64 = 100,
		num_thermal::Int64 = 100,
		measuring_interval::Int64 = 1,
		saving_interval::Int64 = 100,
		w_list::Array{Float64,1} = collect(-3.0:0.1:3.0),
		guess::String = "uniform",
		U_disorder::String = "uniform",
		diag_method::String = "K0",
		pairing_symmetry::String = "s-wave",
		directory::String = ".",
		num_cores::Int64 = 1,
		num_threads::Int64 = 1,
		start_time::DateTime = Dates.now(),
		w_conductivity_range::Array{Float64,1} = collect(-2.0:0.1:2.0),
		delta_values_fixed::Array{Float64,1} = [0.01, 0.05, 0.1, 0.2],
		delta_values_stiffness::Array{Float64,1} = [0.01, 0.05, 0.1, 0.2],
		U_disorder_params::Array{Float64,1} = [0.0, 10.0, 0.7, 0.7],
		U_impurity::Float64 = 1.0,
		TCluster_size_x::Int64 = 2,
		TCluster_size_y::Int64 = TCluster_size_x,  # Default to square TCluster
		smooth_U::Tuple{Bool, Float64} = (false, 0.8),
		loop_random::Bool = true,
		B::Float64 = 0.0,
		gauge_choice::String = "LandauX",
		boundary_conditions::String = "PBC",
		dw_values::Array{Float64,1} = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
		compute_jP::Bool = false,
		disorder_seed::Int64 = 0,   # 0 => derive from `seed` (legacy behaviour)
		alpha::Float64 = 1.0,       # 1 => all U in pairing channel (legacy)
		beta_phi::Float64 = 0.0,    # 0 => density channel OFF (legacy)
		Q_density::Array{Float64,1} = [0.0, 0.0],  # density ordering vector
		snapshot_interval::Int64 = 500,  # raw MC-field snapshot cadence (0 => OFF)
		phi_hist_range::Array{Float64,1} = [-10.0, 10.0],  # phi histogram [lo, hi]
		phi_hist_nbins::Int64 = 100,                       # phi histogram bins (0 => OFF)
		Delta_hist_abs_range::Array{Float64,1} = [0.0, 10.0],  # Delta 2D-hist amplitude axis [lo, hi]
		Delta_hist_abs_nbins::Int64 = 100,                     # Delta 2D-hist amplitude bins (0 => OFF)
		Delta_hist_arg_nbins::Int64 = 50,  # Delta 2D-hist phase bins over [0,2pi] (0 => OFF)
		mu_disorder::String = "none",      # "none" | "interaction-only" | "coupled" | "potential-only"
		mu_disorder_f::Float64 = 0.0       # fraction of sites with U_i=0 (mu unshifted)
	)
		N = Lx * Ly
		SimData(Lx, Ly, N, SC_L, U, t, V, seed, mu, beta, num_measure, num_thermal,
				measuring_interval, saving_interval, w_list, guess, U_disorder,
				diag_method, pairing_symmetry, directory, num_cores, num_threads,
				start_time, w_conductivity_range, delta_values_fixed, delta_values_stiffness, U_disorder_params, U_impurity,
				TCluster_size_x, TCluster_size_y, smooth_U, loop_random, B,
				gauge_choice, boundary_conditions, dw_values, compute_jP, disorder_seed,
				alpha, beta_phi, Q_density, snapshot_interval,
				phi_hist_range, phi_hist_nbins, Delta_hist_abs_range, Delta_hist_abs_nbins,
				Delta_hist_arg_nbins, mu_disorder, mu_disorder_f)
	end

	"""
	    effective_disorder_seed(sim_data) -> Int64

	The seed that actually draws the disorder realization. `disorder_seed == 0`
	means "not specified", which reproduces the legacy behaviour of deriving it
	from the Monte-Carlo seed.
	"""
	effective_disorder_seed(sim_data::SimData) =
		sim_data.disorder_seed == 0 ? sim_data.seed : sim_data.disorder_seed

	"""
	    getproperty(s::SimData, name)

	Field access with two old names kept for old code: `s.L` returns `s.Lx` and
	`s.TCluster_size` returns `s.TCluster_size_x`. Use Lx, Ly, N in new code.
	"""
	function Base.getproperty(s::SimData, name::Symbol)
		if name === :L
			return getfield(s, :Lx)
		elseif name === :TCluster_size
			# Legacy: return TCluster_size_x for backward compatibility
			return getfield(s, :TCluster_size_x)
		else
			return getfield(s, name)
		end
	end
end

	"""
	    ConstStruct

	Everything that stays fixed during a run, built once by `Const_Struct(sim_data)`:
	lattice indices and coordinates, hopping matrices (with boundary twists and Peierls
	phases), the disorder maps (`U_params`, `V_matrix`, `mu_shift`), the travelling-cluster
	site lists, k and q lists with their Fourier tables, and the spectral-function k-path.
	Not saved in checkpoints; rebuilt from `SimData` on resume.
	"""
	mutable struct ConstStruct
		# constants over the simulation
		# Lattice dimensions (stored for convenience)
		Lx::Int64
		Ly::Int64
		N::Int64  # = Lx * Ly
		# CartesianIndices for linear -> (x,y) conversion
		CI::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
		# LinearIndices for (x,y) -> linear conversion
		LI::LinearIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
		# x array
		x::Array{Int64,1}
		# y array
		y::Array{Int64,1}
		# update Distribution
		dist::IsoNormal
		# Nearest Neighbor matrix vector for each supercell k-vector
		NN_mat::Array{Array{ComplexF64,2},1}
		# Nearest Neighbor x direction matrix for d-wave
		NN_x::Array{ComplexF64,2}
		# Nearest Neighbor y direction matrix for d-wave
		NN_y::Array{ComplexF64,2}
		# V_matrix - disorder on each site
		V_matrix::Array{Float64,1}
		# U_parameters_vector - interaction strength
		U_params::Array{Float64,1}
		# U_parameters_vector for x, y direction for d-wave
		U_params_x::Array{Float64,1}
		U_params_y::Array{Float64,1}
		# resolution of the w_list
		w_delta::Float64
		# kx_list - momentum values in x direction
		kx_list::Array{Float64,1}
		# ky_list - momentum values in y direction
		ky_list::Array{Float64,1}
		# k_vectors - list of k-vectors of the real lattice
		k_vectors::Array{Array{Float64,1},1}
		#spins = [1,-1]
		spins::Array{Int64,1}
		#new index list - (x,y) -> (x+1,y)
		new_ix_list::Array{Int64,1}
		# new index list y - (x,y) -> (x,y+1)
		new_iy_list::Array{Int64,1}
		# ls list for Ps_l calculation
		ls_list::Array{Array{Int64, 1},1}
		# reduced ls list - only until L/2
		ls_list_reduced::Array{Array{Int64, 1},1}
		# SF_k_path - path in k-space for SF calculation
		SF_k_path::Array{Array{Int64, 1},1}
		# q_list for current-current correlation (qx=0, qy) and (qx, qy=0)
		q_list_CC::Array{Array{Float64,1},1}
		# q_list for susceptibility (2*pi/Lx * (0:Lx/2), qy=0)
		q_list_sus::Array{Array{Float64,1},1}
		# signatures for quick q-list cache matching
		q_list_CC_sig::UInt64
		q_list_sus_sig::UInt64
		# precomputed FT tables for default q lists (N x n_q)
		FT_vecs_q_CC::Array{ComplexF64,2}
		FT_vecs_q_sus::Array{ComplexF64,2}
		# q_tuples - tuples of q values for Current Current calculations (legacy)
		q_tuples_CC::Array{Tuple{Float64,Float64},1}
		# q_ix_tuples - q index tuples for Current Current calculations (legacy)
		q_ix_tuples_CC::Array{Tuple{Int64,Int64},1}
		# Traveling cluster NN matricies
		TCluster_NN_mats::Array{Array{ComplexF64,2},1}
		# Traveling cluster assignments
		TCluster_assignments::Array{Array{Int64,1},1}
		# SC_Ki_list
		SC_Ki_list::Array{Float64,1}
		# K vectors of the supercell space
		SC_K_vectors::Array{Array{Float64,1},1}
		# munis SC K vectors index
		SC_minus_K::Array{Int64,1}
		# General k-vectors list
		k_general_vectors::Array{Array{Float64,1},1}
		# General k-vectors idx->xy mapping
		k_general_xy::Array{Array{Int64,1},1}
		# General k-vectors xy->idx mapping
		k_general_idx::Array{Int64,2}
		# General k-vectors index to q,K index mapping
		k_general_qK_idx::Array{Array{Int64,1},1}
		# x egdes - 1 for x==L, 0 else
		x_edges::Array{Int64,1}
		# d-wave pairing symmetries
		d_wave_pairings::Array{String,1}
		# s-wave pairing symmetries
		s_wave_pairings::Array{String,1}
		# Peierls phase factors for magnetic field
		peierls_phases::Array{ComplexF64,2}
		# Density-channel modulation sign s_i = cos(Q.r_i) for the phi coupling.
		# +1 everywhere for Q=0 (uniform Hartree), (-1)^(x+y) for Q=(pi,pi) (CDW).
		density_sign::Array{Float64,1}
		# Per-site chemical-potential shift for the coupled/potential mu-U disorder
		# (mu_i = mu - U_i/2). RECORD ONLY: it is already FOLDED INTO V_matrix at
		# construction (both are additive on-site potentials, so the BdG diagonal
		# mu - V_matrix[i] + density_sign[i]*phi[i] already includes it). Kept as a
		# separate field so analysis can see which sites are shifted / verify f.
		# All zeros unless sim_data.mu_disorder != "none".
		mu_shift::Array{Float64,1}
	end

	"""
	    getproperty(c::ConstStruct, name)

	Field access; the old name `c.ki_list` returns `c.kx_list`.
	"""
	function Base.getproperty(c::ConstStruct, name::Symbol)
		if name === :ki_list
			return getfield(c, :kx_list)
		else
			return getfield(c, name)
		end
	end

	# --- work buffers of the fast measurement kernels (make_measurement_K!(...; fast=true)) ---
	# Built once per run by measurement_work! (lazily, on the first full
	# measurement), never checkpointed. See "fast measurement kernels" below.

	"""
	    CurrentVertexGroup

	One lattice row (`axis = :y`, used for q = (0, qy), q = 0 included) or column
	(`axis = :x`, used for q = (qx, 0), qx ≠ 0) of the row-block current-vertex
	method. q-independent.
	"""
	struct CurrentVertexGroup
		axis::Symbol                     # :y -> row y, :x -> column x
		coord::Int64                     # that row's y (column's x); enters e^{i q coord}
		rows::Array{Int64,1}             # rows of U touched by the group's bonds (up, then down)
		Mg::Array{ComplexF64,2}          # the group's current operator restricted to `rows`
	end

	"""
	    CurrentVertexOp

	The q-independent current operator of one direction (+x or +y bonds), in the two
	forms the vertex kernels use: a sparse 2N×2N pattern (per-q GEMM) and row/column
	groups (row-block method).
	"""
	struct CurrentVertexOp
		M::SparseMatrixCSC{ComplexF64,Int64}  # 2N x 2N pattern of j(q) (per-q GEMM); nzval rewritten per q
		slot::Array{Int64,1}                  # term k (4 per site) -> its index in M.nzval
		Tr::Array{ComplexF64,1}               # boundary twist x Peierls phase of the bond i -> i + a
		groups::Array{CurrentVertexGroup,1}   # row / column blocks (row-block method)
	end

	"""
	    MeasurementWork

	Work buffers of the fast measurement kernels (`make_measurement_K!(...; fast=true)`),
	built lazily by `measurement_work!` on the first full measurement and never
	checkpointed. Rebuilt if the sizes it was made for change.
	"""
	struct MeasurementWork
		# what the buffers were sized for (measurement_work! rebuilds on a mismatch)
		N::Int64
		n_w::Int64
		n_w_cond::Int64
		n_delta::Int64
		n_chunks::Int64                  # accumulator chunks of the threaded fixed-delta loop
		# current vertices G^a(q)
		vertex_x::CurrentVertexOp
		vertex_y::CurrentVertexOp
		UR::Array{ComplexF64,2}          # U[rows, :]             (row-block)
		MU::Array{ComplexF64,2}          # Mg * U[rows, :]        (row-block)
		# 2N x 2N complex scratch, shared by kernels that never run at the same time:
		# G_row (row-block), M(q)*U (per-q GEMM), V = U_up' U_dn (DdD), F_all*U_up (BZ)
		scratch::Array{ComplexF64,2}
		# L_ab(q, w=0)
		Mlm::Array{Float64,2}            # (f_l - f_m)/(E_l - E_m); SAME MEMORY as pair_dE
		Lab_q::Array{ComplexF64,2}       # 4 x n_q, rows xx, xy, yx, yy
		# fixed-delta conductivity: packed eigen-pairs and per-chunk accumulators
		pair_dE::Array{Float64,1}        # E_l - E_m  (= vec(Mlm), used after L_ab(q) is done)
		pair_xx::Array{Float64,1}        # df |gx|^2
		pair_yy::Array{Float64,1}        # df |gy|^2
		pair_xy_re::Array{Float64,1}     # real(df gx conj(gy))
		pair_xy_im::Array{Float64,1}     # imag(df gx conj(gy))
		fd_acc::Array{Float64,3}         # 8 x (n_delta * n_w_cond) x n_chunks
		# spectral function on the k-path, BZ map at w = 0, LDOS
		F_path::Array{ComplexF64,2}      # F_path[p, n] = e^{-i k_p . r_n}, k_p on SF_k_path
		F_all::Array{ComplexF64,2}       # same for every k of the BZ map
		Lor::Array{Float64,2}            # Lor[l, j] = (delta/pi) / (delta^2 + (E_l - w_j)^2)
		FU_path::Array{ComplexF64,2}     # F_path * U_up
		W_path::Array{Float64,2}         # |F_path * U_up|^2
		W_N2N::Array{Float64,2}          # N x 2N: |F_all * U_up|^2 (BZ), then |U_up|^2 (LDOS)
		A_path::Array{Float64,2}         # output A(k_p, w_j)
		BZ::Array{Float64,1}             # output A(k, w = 0)
		LDOS::Array{Float64,2}           # output LDOS(i, w_j)
	end

	"""
	    TemporaryData

	Preallocated scratch space reused by every update and measurement: backups of the
	eigen-solution for reverting a rejected update, Hamiltonian and eigen-solver
	workspaces (full lattice and travelling cluster), Green's-function and Fourier
	buffers of the measurements. Built by `Temporary_Data(sim_data)`; not checkpointed.
	"""
	mutable struct TemporaryData
		# temporary data storage

		# Backup energy levels
		Energies_Bckup::Array{Float64,2} # for each SC K (λ,K)
		# Backup diagonalizing maticies U
		U_matrix_Bckup::Array{ComplexF64,3} # for each SC K ((xσ,λ),K)
		# K hamiltonian matrix
		K::Array{ComplexF64,2}
		# Energies vector for current K computation
		Energies_K::Array{Float64,1}
		# U_matrix for current K computation
		U_matrix_K::Array{ComplexF64,2}
		# update suggestion vector (real, imag)
		Delta_suggestion::Array{Float64,1}
		# workspace for diagonalization
		ws::HermitianEigenWs
		# full spectral function
		SF::Array{ComplexF64,1}
		# f - Fermi-dirac distribution function
		f::Array{Float64,2} # for each SC K (λ,K)
		# exp_beta_Energies = e^(beta*Energies)
		exp_beta_Energies::Array{Float64,2} # for each SC K (λ,K)
		# delta - regularization parameter for Lorentzian broadening (mean_diff_low method)
		delta::Float64
		# delta_max - maximum delta for Lorentzian broadening
		delta_max::Float64
		# M  - matrix used in current_current_correlation => M = (e^(beta*En_i) - e^(beta*En_j)) / (iw_n +En_i - En_j)
		M::Array{ComplexF64,2}
		# v_i - vector used in current_current_correlation *** used as allocation for g0 in c-c-corr
		v_i::Array{ComplexF64,1}
		# S_r martix 
		S_r::Array{ComplexF64,2}
		# S_r_vec 
		S_r_vec::Array{ComplexF64,1}
		# combined real-space G_r scratch (up - down_swapped)
		G_r_vec::Array{ComplexF64,1}
		# preallocated FT vectors for q-batched G construction (N x q_capacity)
		FT_vecs_q::Array{ComplexF64,2}
		# preallocated q-space projection scratch (length q_capacity)
		G_q_proj::Array{ComplexF64,1}
		# S_q matrix
		S_q::Array{ComplexF64,2}
		# G_matrix used in current-current correlation calculations
		G_matrix::Array{ComplexF64,2}
		# g_vec used in Delta operator computation
		g_A::Array{ComplexF64,1}
		g_B::Array{ComplexF64,1}
		# fft plan for S_r = NxN matrix
		plan_S_r::FFTW.cFFTWPlan
		# S_up/down for x and y directions
		S_up_x::Array{Array{ComplexF64,2},1}
		S_down_x::Array{Array{ComplexF64,2},1}
		S_up_y::Array{Array{ComplexF64,2},1}
		S_down_y::Array{Array{ComplexF64,2},1}
		# G_matrix_vec for x and y directions. Empty until the first measurement
		# needs them (ensure_G_buffers! fills these same vectors in place).
		G_matrix_vec_x::Array{Array{ComplexF64,2},1}
		G_matrix_vec_y::Array{Array{ComplexF64,2},1}
		# LDOS temp array
		LDOS_vec::Array{Float64,1}
		# Traveling cluster K matrix
		TCluster_K::Array{ComplexF64,2}
		# Traveling cluster V matrix
		TCluster_V::Array{Float64,1}
		# Traveling cluster Delta matrix
		TCluster_Delta_s::Array{ComplexF64,1}
		TCluster_Delta_dx::Array{ComplexF64,1}
		TCluster_Delta_dy::Array{ComplexF64,1}
		# Traveling cluster density-channel field and its modulation sign
		TCluster_phi::Array{Float64,1}
		TCluster_dsign::Array{Float64,1}
		# Traveling Cluster ws
		TCluster_ws::HermitianEigenWs
		# Tcluster temp energies
		TCluster_energies::Array{Float64,1}
		# Tcluster temp U_matrix
		TCluster_U_matrix::Array{ComplexF64,2}
		# temp array current-current correlation
		Lxx_temp::Array{ComplexF64,1}
		# susceptibility workspace (row-weighted up and row-weighted conjugated down blocks)
		sus_weighted_up::Array{ComplexF64,2}
		sus_weighted_dn_conj::Array{ComplexF64,2}
		# work buffers of the fast measurement kernels; nothing until the first full
		# measurement (the n_only mu-search never builds them)
		meas_work::Union{Nothing, MeasurementWork}

	end



begin #structs and measurments

	"""
	    MeasureData

	The measured streams of a run: one entry per measurement for the last
	`saving_interval` measurements (dumped to results.jld2 in chunks, see `dump_measurements`).
	`A`, `BZ` and `LDOS` are running means, not streams. The comment on each field says
	what it holds; `get_means` (sim_num_control.jl) turns them into means ± errors.
	"""
	mutable struct MeasureData
		#MCMF action
		Ss::Array{Float64,1}
		# reduced spectral function 
		A::Array{Array{Float64,1},1}
		# Brillouin zone - A(w=0) (only mean)
		BZ::Array{Float64,1}
		#desity of states for w_list
		DOS::Array{Array{Float64,1},1}
		#avrage occupation : n
		n::Array{Float64,1}
		# Spin susceptibility chi_s(q)
		chi_spin::Array{Array{Float64,1},1}
		# Charge susceptibility chi_c(q)
		chi_charge::Array{Array{Float64,1},1}
		# Kx - kinetic energy observable in x direction
		Kx::Array{Float64,1}
		# Ky - kinetic energy observable in y direction
		Ky::Array{Float64,1}
		# Lxx(qx=0, qy->0, w=0) - current-current correlation in x direction as a function of qy with qx,w=0.
		Lxx_qy::Array{Array{Float64,1},1}
		# Lxx(qx->0, qy=0, w=0) - current-current correlation in x direction as a function of qx with qy,w=0.
		Lxx_qx::Array{Array{Float64,1},1}
		# Lxy(qx=0, qy->0, w=0) - current-current correlation xy as a function of qy with qx,w=0.
		Lxy_qy::Array{Array{ComplexF64,1},1}
		# Lxy(qx->0, qy=0, w=0) - current-current correlation xy as a function of qx with qy,w=0.
		Lxy_qx::Array{Array{ComplexF64,1},1}
		# Lyx(qx=0, qy->0, w=0) - current-current correlation yx as a function of qy with qx,w=0.
		Lyx_qy::Array{Array{ComplexF64,1},1}
		# Lyx(qx->0, qy=0, w=0) - current-current correlation yx as a function of qx with qy,w=0.
		Lyx_qx::Array{Array{ComplexF64,1},1}
		# Lyy(qx=0, qy->0, w=0) - current-current correlation in y direction as a function of qy with qx,w=0.
		Lyy_qy::Array{Array{Float64,1},1}
		# Lyy(qx->0, qy=0, w=0) - current-current correlation in y direction as a function of qx with qy,w=0.
		Lyy_qx::Array{Array{Float64,1},1}
		# Lab(q=0, w=0) - current-current correlation tensor at q=0, w=0 (one scalar per measurement step)
		Lxx_q0::Array{Float64,1}
		Lxy_q0::Array{ComplexF64,1}
		Lyx_q0::Array{ComplexF64,1}
		Lyy_q0::Array{Float64,1}
		# Delta_operator - Δ=∑ c↑ c↓ operator
		Delta_operator::Array{ComplexF64,1}
		# Delta_mean - avrage of the delta field
		Delta_mean::Array{ComplexF64,1}
		# DdD - <Delta^dagger * Delta>
		DdD::Array{Float64,1}
		# Ps_l - <Δ_i * Δ_i+l^dagger> summed over i, for each l in ls_list (see compute_Ps_l_K)
		Ps_l::Array{Array{Float64,1},1}
		# LDOS(i,w) aray of [[LDOS(i,w) for i] for w]
		LDOS::Array{Array{Float64,1},1}
		# Phase correlation function <e^(i(phi_i-phi_j))>
		Phase_Corr::Array{Array{Float64,1},1}
		# U4: holds the K0 action S_K0 (old name; reported as "K0_action" by get_means)
		U4::Array{Float64,1}
		# D-wave phase correlation - not used in s-wave
		Phase_Corr_dwave_x::Array{Array{Float64,1},1}
		Phase_Corr_dwave_y::Array{Array{Float64,1},1}
		# mixed d-wave phase correlation - not used in s-wave 
		Phase_Corr_mixed::Array{Array{Float64,1},1}
		# Delta_MC_abs = |Delta_d_i| (s than dx than dy)
		Delta_MC_abs::Array{Array{Float64,1},1}
		# Delta_MC = <Delta_d>_i (s than dx than dy)
		Delta_MC::Array{Array{ComplexF64,1},1}
		
		# === Fixed-delta conductivity observables ===
		# These use constant delta values (from sim_data.delta_values_fixed) instead of dynamic mean_diff_low
		# Shape: [saving_interval][n_deltas][n_w] - extra dimension for different constant delta values
		# Lxx_w with fixed delta values
		Lxx_w_fixed::Array{Array{Array{ComplexF64,1},1},1}
		# Lxy_w with fixed delta values
		Lxy_w_fixed::Array{Array{Array{ComplexF64,1},1},1}
		# Lyx_w with fixed delta values
		Lyx_w_fixed::Array{Array{Array{ComplexF64,1},1},1}
		# Lyy_w with fixed delta values
		Lyy_w_fixed::Array{Array{Array{ComplexF64,1},1},1}
		
		# === Energy spacing observables ===
		# Mean of all energy spacings (after filtering numerical noise)
		energy_spacing_mean::Array{Float64,1}
		# Delta computed via mean_diff_low method (mean of small energy spacings)
		delta_mean_diff_low::Array{Float64,1}
		# === DC conductivity (delta->0 integrated formula) ===
		# Shape: [saving_interval][n_dw_values] - one value per fixed Δω window
		conductivity_dc_xx::Array{Array{Float64,1},1}
		conductivity_dc_xy::Array{Array{Float64,1},1}
		conductivity_dc_yx::Array{Array{Float64,1},1}
		conductivity_dc_yy::Array{Array{Float64,1},1}
		# === L_D DC (delta->0 integrated current-current correlator) ===
		# Shape: [saving_interval][n_dw_values] - one value per fixed Δω window
		L_D_dc_xx::Array{Array{Float64,1},1}
		L_D_dc_xy::Array{Array{Float64,1},1}
		L_D_dc_yx::Array{Array{Float64,1},1}
		L_D_dc_yy::Array{Array{Float64,1},1}
		# === FT of unperturbed paramagnetic current density ===
		# <j^P_a(q)>_0 from "Kubo formula with magnetic field" note; q-grid matches Lxx_q* (q_list_CC).
		jPx_q0::Array{ComplexF64,1}
		jPx_qy::Array{Array{ComplexF64,1},1}
		jPx_qx::Array{Array{ComplexF64,1},1}
		jPy_q0::Array{ComplexF64,1}
		jPy_qy::Array{Array{ComplexF64,1},1}
		jPy_qx::Array{Array{ComplexF64,1},1}
		# === density-channel (phi) observables ===
		# phi_mean = (1/N) sum_i phi_i  (spatial mean of the density field)
		phi_mean::Array{Float64,1}
		# phi_abs_mean = (1/N) sum_i |phi_i|
		phi_abs_mean::Array{Float64,1}
		# phi_Q = (1/N) sum_i cos(Q.r_i) phi_i  -> density-channel order amplitude
		# (uniform Hartree component for Q=0, staggered CDW amplitude for Q=(pi,pi))
		phi_Q::Array{Float64,1}
		# m_cdw = (1/N) sum_i cos(Q.r_i) <n_i>  -> electron charge-density-wave order
		# parameter. Only computed for Q != 0 (for Q=0 it equals the density n).
		m_cdw::Array{Float64,1}
		# === L_D direct (ω→0, δ→0 closed-form current-current correlator) ===
		# Re[Λ_aa'(q=0, ω→0, δ→0)] computed directly (no ω, no δ), one real scalar
		# per measurement. Off-diagonal (xy, yx) use Re[G^a (G^a')*], NOT |G|^2.
		# Shape: [saving_interval].
		L_D_direct_xx::Array{Float64,1}
		L_D_direct_xy::Array{Float64,1}
		L_D_direct_yx::Array{Float64,1}
		L_D_direct_yy::Array{Float64,1}
	end

	"""
	    Initial_measurments(sim_data) -> MeasureData

	Allocate empty measurement buffers for `saving_interval` measurements. Optional
	observables (d-wave phase correlations, the paramagnetic current jP) get empty
	arrays when they are off.
	"""
	function Initial_measurments(sim_data::SimData)
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		SC_L = sim_data.SC_L
		Lx_gen = Lx * SC_L
		Ly_gen = Ly * SC_L
		L_max = max(Lx, Ly)
		L_min = min(Lx, Ly)
		L_gen_max = max(Lx_gen, Ly_gen)
		saving_interval = sim_data.saving_interval
		# q_list for susceptibility: 2*pi/Lx*(0:Lx/2)
		q_list_sus = [[2*pi/sim_data.Lx * q, 0.0] for q in 0:div(sim_data.Lx,2)]
		n_q_sus = length(q_list_sus)
		
		# Size of ls_list_reduced: L_min/2 (diagonal) + L_max/2 + 1 (horizontal)
		ls_list_reduced_size = div(L_min, 2) + div(L_max, 2) + 1
		measure_dwave_observables = sim_data.pairing_symmetry in ["d-wave", "d-p-wave", "mayr-d-wave", "d+s-wave"]
		compute_jP = sim_data.compute_jP

		# Spectral function k-path length: seg1 + seg2 + seg3 + 1 (final (0,0) point)
		# seg1: Lx_gen - div(Lx_gen,2), seg2: Ly_gen - div(Ly_gen,2), seg3: Lx_gen - div(Lx_gen,2)
		SF_path_len = 2 * (Lx_gen - div(Lx_gen, 2)) + (Ly_gen - div(Ly_gen, 2)) + 1

		MeasureData(zeros(saving_interval), # Ss
					[zeros(SF_path_len) for w in sim_data.w_list], #A (spectral function path length)
					zeros(Float64, Lx_gen * Ly_gen), # BZ
					[zeros(length(sim_data.w_list)) for i in 1:saving_interval], # DOS
					zeros(saving_interval),# n
					[zeros(Float64, n_q_sus) for _ in 1:saving_interval], # chi_spin
					[zeros(Float64, n_q_sus) for _ in 1:saving_interval], # chi_charge
					zeros(saving_interval), # Kx
					zeros(saving_interval), # Ky
					[zeros(Float64, div(Ly_gen, 2)) for i in 1:saving_interval], # Lxx_qy
					[zeros(Float64, div(Lx_gen, 2)) for i in 1:saving_interval], # Lxx_qx
					[zeros(ComplexF64, div(Ly_gen, 2)) for i in 1:saving_interval], # Lxy_qy
					[zeros(ComplexF64, div(Lx_gen, 2)) for i in 1:saving_interval], # Lxy_qx
					[zeros(ComplexF64, div(Ly_gen, 2)) for i in 1:saving_interval], # Lyx_qy
					[zeros(ComplexF64, div(Lx_gen, 2)) for i in 1:saving_interval], # Lyx_qx
					[zeros(Float64, div(Ly_gen, 2)) for i in 1:saving_interval], # Lyy_qy
					[zeros(Float64, div(Lx_gen, 2)) for i in 1:saving_interval], # Lyy_qx
					zeros(Float64, saving_interval),    # Lxx_q0
					zeros(ComplexF64, saving_interval), # Lxy_q0
					zeros(ComplexF64, saving_interval), # Lyx_q0
					zeros(Float64, saving_interval),    # Lyy_q0
					zeros(ComplexF64, saving_interval), # Delta_operator
					zeros(ComplexF64, saving_interval), # Delta_mean
					zeros(Float64, saving_interval), # DdD
					[zeros(Float64, ls_list_reduced_size) for i in 1:saving_interval], # Ps_l ([l,l] , [l,0])
					[zeros(Float64, N) for _ in sim_data.w_list], # LDOS
					[zeros(Float64, ls_list_reduced_size) for i in 1:saving_interval], # phase correlation function
					zeros(Float64, saving_interval) # U4
					, (measure_dwave_observables ? [zeros(Float64, ls_list_reduced_size) for i in 1:saving_interval] : Array{Float64,1}[]) # d-wave phase correlation function x direction
					, (measure_dwave_observables ? [zeros(Float64, ls_list_reduced_size) for i in 1:saving_interval] : Array{Float64,1}[]) # d-wave phase correlation function y direction
					, (measure_dwave_observables ? [zeros(Float64, ls_list_reduced_size) for i in 1:saving_interval] : Array{Float64,1}[]) # mixed d-wave phase correlation function
					, [zeros(Float64, measure_dwave_observables ? 3 : 1) for _ in 1:saving_interval]# Delta_MC_abs
					, [zeros(ComplexF64, measure_dwave_observables ? 3 : 1) for _ in 1:saving_interval] # Delta_MC
					# Fixed-delta conductivity observables - shape: [saving_interval][n_deltas][n_w]
					, [[zeros(ComplexF64, length(sim_data.w_conductivity_range)) for _ in eachindex(sim_data.delta_values_fixed)] for _ in 1:saving_interval] # Lxx_w_fixed
					, [[zeros(ComplexF64, length(sim_data.w_conductivity_range)) for _ in eachindex(sim_data.delta_values_fixed)] for _ in 1:saving_interval] # Lxy_w_fixed
					, [[zeros(ComplexF64, length(sim_data.w_conductivity_range)) for _ in eachindex(sim_data.delta_values_fixed)] for _ in 1:saving_interval] # Lyx_w_fixed
					, [[zeros(ComplexF64, length(sim_data.w_conductivity_range)) for _ in eachindex(sim_data.delta_values_fixed)] for _ in 1:saving_interval] # Lyy_w_fixed
					# Energy spacing observables
					, zeros(Float64, saving_interval) # energy_spacing_mean
					, zeros(Float64, saving_interval) # delta_mean_diff_low
					# DC conductivity (delta->0 integrated formula)
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # conductivity_dc_xx
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # conductivity_dc_xy
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # conductivity_dc_yx
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # conductivity_dc_yy
					# L_D DC (delta->0 integrated current-current correlator)
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # L_D_dc_xx
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # L_D_dc_xy
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # L_D_dc_yx
					, [zeros(Float64, length(sim_data.dw_values)) for _ in 1:saving_interval] # L_D_dc_yy
					# FT paramagnetic current (q_list_CC grid); empty when compute_jP is off
					, (compute_jP ? zeros(ComplexF64, saving_interval) : ComplexF64[]) # jPx_q0
					, (compute_jP ? [zeros(ComplexF64, div(Ly_gen, 2)) for i in 1:saving_interval] : Array{ComplexF64,1}[]) # jPx_qy
					, (compute_jP ? [zeros(ComplexF64, div(Lx_gen, 2)) for i in 1:saving_interval] : Array{ComplexF64,1}[]) # jPx_qx
					, (compute_jP ? zeros(ComplexF64, saving_interval) : ComplexF64[]) # jPy_q0
					, (compute_jP ? [zeros(ComplexF64, div(Ly_gen, 2)) for i in 1:saving_interval] : Array{ComplexF64,1}[]) # jPy_qy
					, (compute_jP ? [zeros(ComplexF64, div(Lx_gen, 2)) for i in 1:saving_interval] : Array{ComplexF64,1}[]) # jPy_qx
					# density-channel (phi) observables
					, zeros(Float64, saving_interval) # phi_mean
					, zeros(Float64, saving_interval) # phi_abs_mean
					, zeros(Float64, saving_interval) # phi_Q
					, zeros(Float64, saving_interval) # m_cdw (electron staggered density)
					# L_D direct (ω→0, δ→0 closed-form current-current correlator)
					, zeros(Float64, saving_interval) # L_D_direct_xx
					, zeros(Float64, saving_interval) # L_D_direct_xy
					, zeros(Float64, saving_interval) # L_D_direct_yx
					, zeros(Float64, saving_interval) # L_D_direct_yy
					)
	end


	"""
	    CurrentState

	The Monte-Carlo state: the fields (Δ_s, Δ_dx, Δ_dy, φ), the action and the current
	eigen-solution, step and measurement counters, timing, and the field histograms.
	This is what a checkpoint saves.
	"""
	mutable struct CurrentState
		# delta : HS field
		# s-wave pairing symmetry
		Delta_s::Array{ComplexF64,1}
		# d-wave pairing symmetry
		Delta_dx::Array{ComplexF64,1}
		Delta_dy::Array{ComplexF64,1}
		# phi : density-channel HS field (real, on-site). Couples as a local
		# chemical-potential shift mu -> mu + density_sign_i * phi_i.
		phi::Array{Float64,1}
		#MCMF action
		S::Float64
		# Action per supercell K=0
		S_K0::Float64
		#Energy levels - per supercell K
		Energies::Array{Float64,2}
		#Diagonalizing maticies U - per supercell K
		U_matrix::Array{ComplexF64,3}
		# True when S/Energies/U_matrix are synchronized with current Delta fields
		diag_is_synced::Bool
		# step_count value at the last full synchronization
		diag_synced_step::Int64
		# counter of number of MC steps taken
		step_count::Int64
		# current number of measurments
		num_current_measurments::Int64
		# index of current measurement in MeasureData
		saving_measurement_ix::Int64
		# Last dump time
		last_dump_time::DateTime
		# Cumulative wall-time spent inside MC_stepV2! calls
		timing_mc_step_total_sec::Float64
		# Number of MC_stepV2! calls timed
		timing_mc_step_count::Int64
		# Cumulative wall-time spent inside make_measurement_K! calls
		timing_measurement_total_sec::Float64
		# Number of measurement calls timed
		timing_measurement_count::Int64
		# --- MC-field histograms: lifetime click-count accumulators (constant memory,
		# never reset). Filled once per measurement over all sites; out-of-range
		# values are clamped into the edge bins. Sizes come from the matching sim_data
		# *_nbins (0-length when OFF). Bin ranges/counts live in sim_data. Sits in
		# CurrentState so it is checkpointed and resumes correctly.
		phi_hist::Array{Int64,1}   # 1D: phi_i over sim_data.phi_hist_range
		# 2D JOINT histogram of (|Delta_s_i|, arg(Delta_s_i)): rows = amplitude bins
		# (sim_data.Delta_hist_abs_*), cols = phase bins over [0, 2pi]. Empty (0x0) when OFF.
		Delta_hist::Array{Int64,2}
	end


	"""
	    mf_uniform_guess(sim_data; U_for_guess) -> (Delta_mag, phi_val)

	Homogeneous k-space mean-field seed for the "uniform" initial guess. For the
	s-wave, Q=(0,0) case it solves the COUPLED Delta+phi saddle at the sim's mu
	(clean, uniform) and returns the pairing magnitude and the uniform density
	field to seed the MC with. For d-wave or Q!=0 (which the k-space solver does
	not cover) it falls back to the legacy symmetry-aware pairing saddle with no
	density seed (phi=0). Deterministic (no RNG).
	"""
	function mf_uniform_guess(sim_data::SimData; U_for_guess::Float64)
		Qx, Qy = sim_data.Q_density[1], sim_data.Q_density[2]
		is_Q0 = abs(Qx) < 1e-12 && abs(Qy) < 1e-12
		if !(is_Q0 && sim_data.pairing_symmetry == "s-wave" && sim_data.alpha > 0.0)
			# Legacy pairing-only seed (handles d-wave via saddle_point_delta); no phi.
			d = saddle_point_delta(sim_data; d_min=0.001, U_params=U_for_guess)[1]
			return (Float64(d), 0.0)
		end
		res = solve_mf(; t=sim_data.t, U=U_for_guess, alpha=sim_data.alpha,
						 beta_phi=sim_data.beta_phi, beta=sim_data.beta,
						 Lx=sim_data.Lx, Ly=sim_data.Ly, mu=sim_data.mu,
						 Delta0=0.5, phi0=0.0, tol=1e-10, mixing=0.5)
		return (res.Delta, res.phi)
	end

	"""
	    mf_starting_mu(sim_data, target_n) -> Union{Float64, Nothing}

	Mean-field estimate of the chemical potential that yields `target_n` in the
	clean uniform system (density channel included, Q=0). Returns `nothing` for
	Q!=0 or an out-of-range target so callers keep the legacy mu bracket. Used to
	seed a tight bracket for occupation convergence instead of the blind (-4, 0).
	"""
	function mf_starting_mu(sim_data::SimData, target_n::Float64)
		Qx, Qy = sim_data.Q_density[1], sim_data.Q_density[2]
		(abs(Qx) < 1e-12 && abs(Qy) < 1e-12) || return nothing   # k-space MF: Q=0 only
		(0.0 < target_n < 2.0) || return nothing
		sim_data.alpha > 0.0 || return nothing
		res = solve_mf(; t=sim_data.t, U=sim_data.U, alpha=sim_data.alpha,
						 beta_phi=sim_data.beta_phi, beta=sim_data.beta,
						 Lx=sim_data.Lx, Ly=sim_data.Ly, target_n=target_n,
						 Delta0=0.5, tol=1e-10)
		return res.mu_converged ? res.mu : nothing
	end

	"""
	    Initial_state(sim_data; const_data=Const_Struct(sim_data)) -> CurrentState

	Starting field configuration, by `sim_data.guess`:
	- `"rand"`: real and imaginary parts of Δ uniform in [0, 1] on every site;
	- `"uniform"`: |Δ| = Δ_MF·(0.75…1.25) + (0…0.1) with a random phase, where Δ_MF is
	  the clean mean-field gap (`mf_uniform_guess`); φ starts at its mean-field value.
	Only the fields of `pairing_symmetry` are set (the others are 0), and every field is
	0 on sites where its U is 0. The action and eigen-solution are filled in later
	(`Lattice_Data`).
	"""
	function Initial_state(sim_data::SimData; const_data::ConstStruct=nothing)

		#initial delta
        guess = sim_data.guess
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		SC_L = sim_data.SC_L
		SC_N = SC_L^2
		pairing = sim_data.pairing_symmetry
		if const_data === nothing
			const_data = Const_Struct(sim_data)
		end
		mean_U = maximum([mean(const_data.U_params), mean(const_data.U_params_x), mean(const_data.U_params_y)])

		# density-channel mean-field seed (stays 0 unless the uniform guess sets it)
		phi_guess = 0.0
		if guess == "rand"
			D1 = rand(N)+ 1im*rand(N)
			D2 = rand(N)+ 1im*rand(N)
			D3 = rand(N)+ 1im*rand(N)

		elseif guess == "uniform"
			# k-space mean field: seed the pairing magnitude AND the density field
			# phi from the coupled MF saddle (s-wave, Q=0); falls back to the
			# pairing-only saddle with phi=0 for d-wave or Q!=0. See mf_uniform_guess.
			UniformGuess, phi_guess = mf_uniform_guess(sim_data; U_for_guess=mean_U)
			D1 =  (UniformGuess *((0.75 .+ 0.5*rand(N))) .+ 0.1*rand(N)) .* exp.(1im*rand(N)*2*pi)
			D2 =  (UniformGuess *((0.75 .+ 0.5*rand(N))) .+ 0.1*rand(N)) .* exp.(1im*rand(N)*2*pi)
			D3 =  (UniformGuess *((0.75 .+ 0.5*rand(N))) .+ 0.1*rand(N)) .* exp.(1im*rand(N)*2*pi)
		end
		if pairing == "s-wave"
			Delta_s = D1
			Delta_dx = zeros(ComplexF64, N)
			Delta_dy = zeros(ComplexF64, N)
		elseif pairing == "d-wave"
			Delta_s = zeros(ComplexF64, N)
			Delta_dx = D2
			Delta_dy = D3
		elseif pairing == "mayr-d-wave"
			Delta_s = zeros(ComplexF64, N)
			Delta_dx = D2
			Delta_dy = abs.(D2) .* exp.(1im*rand(N)*2*pi)
		elseif pairing == "d-p-wave"
			Delta_s = zeros(ComplexF64, N)
			Delta_dx = D2
			Delta_dy = - D2
		elseif pairing == "d+s-wave"
			Delta_s = D1
			Delta_dx = D2
			Delta_dy = D3
		else
			error("Pairing symmetry $(pairing) not recognized.")
		end

		comp_zero = 0.0 + 0.0im

		# phi: density-channel field, seeded uniformly at the mean-field value
		# (0 for the legacy/off channel and for the rand guess). The Monte-Carlo
		# sweep evolves it from here when the density channel is active.
		phi = fill(phi_guess, N)

		for i in 1:N
			if const_data.U_params[i] == 0.0
				Delta_s[i] = comp_zero
				phi[i] = 0.0   # no interaction on this site -> no density field
			end
			if const_data.U_params_x[i] == 0.0
				Delta_dx[i] = comp_zero
			end
			if const_data.U_params_y[i] == 0.0
				Delta_dy[i] = comp_zero
			end
		end


		CurrentState(Delta_s, # Delta_s
					 Delta_dx, # Delta_dx
					 Delta_dy, # Delta_dy
					 phi, # phi (density-channel field)
					 0.0, # S_total
					 0.0, # S_K0
					 zeros(Float64, 2*N, SC_N), # Energies
					 zeros(ComplexF64, 2*N, 2*N, SC_N), # U_matrix
					 false, # diag_is_synced
					 -1, # diag_synced_step
					 -1, # step_count
					 0, # num_current_measurments
					 0, # saving_measurement_ix
					 Dates.now(), # last_dump_time
					 0.0, # timing_mc_step_total_sec
					 0, # timing_mc_step_count
					 0.0, # timing_measurement_total_sec
					 0, # timing_measurement_count
					 zeros(Int64, max(0, sim_data.phi_hist_nbins)),  # phi_hist (1D)
					 # Delta_hist (2D |Delta| x arg): empty 0x0 unless BOTH axes are on
					 (sim_data.Delta_hist_abs_nbins > 0 && sim_data.Delta_hist_arg_nbins > 0 ?
						zeros(Int64, sim_data.Delta_hist_abs_nbins, sim_data.Delta_hist_arg_nbins) :
						zeros(Int64, 0, 0))
					 )
	end
	
	"""
	    LatticeData

	One simulation: parameters (`sim_data`), MC state (`current_state`), measurement
	buffers (`measurement_data`), constants (`const_struct`) and scratch space
	(`temporary_data`). Create it with `Lattice_Data(sim_data)`.
	"""
	mutable struct LatticeData
	    # simulation data
	    sim_data::SimData
		# current system state
		current_state::CurrentState
		#measurments
		measurement_data::MeasureData
		# constent over the simulation - x, y, ix, NN_mat, w_delta,...
		const_struct::ConstStruct
		# temporary data storage
		temporary_data::TemporaryData
	end

	"""
	    Lattice_Data(sim_data) -> LatticeData

	Build a new simulation: constants, initial fields, empty measurements, scratch
	space; then diagonalize once to set the action and eigen-solution of the start
	configuration (and S_K0 for `"K0"`).
	"""
	function Lattice_Data(sim_data::SimData)

		const_struct = Const_Struct(sim_data) # make constants struct
		current_state = Initial_state(sim_data; const_data=const_struct) # make initial state

		#make struct
		data = LatticeData(sim_data, # SimData
					current_state, # CurrentState
					Initial_measurments(sim_data), # MeasureData
					const_struct, #constants struct
					Temporary_Data(sim_data), # temporary data
					)

		# diagonalizing the hamiltonian
		Delta_s = current_state.Delta_s
		Delta_dx = current_state.Delta_dx
		Delta_dy = current_state.Delta_dy
		S, energies, U_matrix = MCMF_action_K(data, Delta_s, Delta_dx, Delta_dy)
		Lattice_update!(data, Delta_s, Delta_dx, Delta_dy, S, energies, U_matrix)
		if sim_data.diag_method == "K0"
			S0, _, _ = MCMF_action_K0(data, Delta_s, Delta_dx, Delta_dy)
			data.current_state.S_K0 = S0
		end

		return data
	end




	"""
	    Const_Struct(sim_data) -> ConstStruct

	Build the run constants (see `ConstStruct`). The disorder maps U_i, V_i and the μ–U
	partition are drawn from the disorder seed (`effective_disorder_seed`); afterwards the
	global random generator is reseeded from the clock (`get_random_seed!`) for the MC.
	Warns once if the pairing symmetry is not s-wave.
	"""
	function Const_Struct(sim_data::SimData)
		Random.seed!(sim_data.seed) # set the seed for random number generator
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		L = Lx  # Legacy: for SC_L calculations, assume square supercell based on Lx
		SC_L = sim_data.SC_L
		Lx_gen = Lx * SC_L
		Ly_gen = Ly * SC_L
		
		# Create CartesianIndices and LinearIndices for efficient indexing
		CI = make_CartesianIndices(Lx, Ly)
		LI = make_LinearIndices(Lx, Ly)

		# Disorder realizations are drawn from the disorder seed, NOT the MC seed,
		# so several MC seeds can sample the same disordered sample.
		d_seed = effective_disorder_seed(sim_data)

		U_params, U_params_x, U_params_y = make_U_params(sim_data.U, Lx, Ly, sim_data.U_disorder; impurity=sim_data.U_impurity, pairing_symmetry=sim_data.pairing_symmetry, U_puddle_params=sim_data.U_disorder_params, seed=d_seed, smooth=sim_data.smooth_U[1], smooth_sigma=sim_data.smooth_U[2]) # U parameters vector

		V_matrix = make_V_matrix(Lx, Ly, sim_data.V; seed=d_seed)

		# Coupled / potential mu-U disorder (mu_i = mu - U_i/2). For "coupled" this
		# also rewrites U_params to the binary {0, U} interaction. mu_shift is FOLDED
		# INTO V_matrix (both are additive on-site potentials), so the BdG diagonal
		# and the TCluster path need no change; mu_shift is stored only for the record.
		mu_shift = build_mu_disorder!(U_params, sim_data; seed=d_seed)
		V_matrix .+= mu_shift

		get_random_seed!()

		w_delta = 4/N # avarage level spacing resolution
		spins = [1, -1]  # spin up and down
		dist = Distributions.MvNormal([0,0], 0.05*maximum([mean(U_params), mean(U_params_x), mean(U_params_y)])) # distribution for update suggestion


		SC_Ki_list = 2*pi/(SC_L*Lx) * [0:SC_L-1;]  # Note: using Lx for SC calculations
		# For rectangular: use different factors for kx and ky
		SC_K_vectors = [[2*pi/(SC_L*Lx)*kx, 2*pi/(SC_L*Ly)*ky] for kx in 0:SC_L-1 for ky in 0:SC_L-1]
		SC_minus_K = [minus_K_ix(k_vec, SC_L) for k_vec in 1:(SC_L^2)]

		pierls_phases = make_peierls_phases(sim_data)

		NN_mat = neighbor_matrix_K_vector(Lx, Ly, SC_K_vectors; B=sim_data.B, gauge_choice=sim_data.gauge_choice) # Nearest Neighbor matrix vector for each supercell k-vector
		NN_x = neighbor_matrix_x(Lx, Ly, "x")
		NN_y = neighbor_matrix_x(Lx, Ly, "y")


		# Traveling Cluster: rectangular cluster support
		TCluster_size_x = sim_data.TCluster_size_x
		TCluster_size_y = sim_data.TCluster_size_y
		TCluster_N = TCluster_size_x * TCluster_size_y
		# Separate scaling factors for x and y directions in SC_K_vectors
		TCluster_SC_K_vectors = [[K[1] * (Lx / TCluster_size_x), K[2] * (Ly / TCluster_size_y)] for K in SC_K_vectors]
		Tcluster_NN_matrix = neighbor_matrix_K_vector(TCluster_size_x, TCluster_size_y, TCluster_SC_K_vectors; B=sim_data.B, gauge_choice=sim_data.gauge_choice)
		TCluster_assignments = [zeros(Int64, TCluster_N) for _ in 1:N] # preallocate TCluster assignments
		
		x_array = zeros(Int64, N)
		y_array = zeros(Int64, N)
		new_ix_list = zeros(Int64, N)
		new_iy_list = zeros(Int64, N)
		xedges = zeros(Int64, N)
		for ix in 1:N
			X = get_x(ix, CI)
			Y = get_y(ix, CI)
			x_array[ix] = X
			y_array[ix] = Y
			xedges[ix] = X == Lx ? 1 : 0
			new_ix_list[ix] = idx(add_xy_ix(X, 1, Lx), add_xy_ix(Y, 0, Ly), Lx)
			new_iy_list[ix] = idx(add_xy_ix(X, 0, Lx), add_xy_ix(Y, 1, Ly), Lx)
			# TCluster assignments (rectangular cluster)
			x_start = add_xy_ix(X, -div(TCluster_size_x, 2), Lx)
			y_start = add_xy_ix(Y, -div(TCluster_size_y, 2), Ly)
			count = 1
			for dy in 0:(TCluster_size_y - 1)
				for dx in 0:(TCluster_size_x - 1)
					xp = add_xy_ix(x_start, dx, Lx)
					yp = add_xy_ix(y_start, dy, Ly)
					ixp = idx(xp, yp, Lx)
					TCluster_assignments[ix][count] = ixp
					count += 1
				end
			end
		end

		# ls_list: for rectangular, use min(Lx, Ly) for diagonal, max for axis
		L_min = min(Lx, Ly)
		L_max = max(Lx, Ly)
		ls_list = [[l,l] for l in div(L_min*SC_L, 2):-1:1]
		append!(ls_list, [[l, 0] for l in 0:div(L_max*SC_L, 2)]) 

		ls_list_reduced = [[l,l] for l in div(L_min, 2):-1:1]
		append!(ls_list_reduced, [[l, 0] for l in 0:div(L_max, 2)]) 

		# Preallocate q_list for current-current correlation: (qx=0, qy) and (qx, qy=0)
		qs_y = (2*pi/Ly) * (1:(div(Ly,2)))
		qs_x = (2*pi/Lx) * (1:(div(Lx,2)))
		q_list_CC = [[0.0, 0.0]]
		append!(q_list_CC, [[0.0, q] for q in qs_y])
		append!(q_list_CC, [[q, 0.0] for q in qs_x])
		# Preallocate q_list for susceptibility: 2*pi/Lx*(0:Lx/2), qy=0
		q_list_sus = [[2*pi/Lx * q, 0.0] for q in 0:div(Lx,2)] # old (overwritten on the next line)
		q_list_sus = [[0.0, 0.0]] # new - we are only interested in q=0
		q_list_CC_sig = q_list_signature(q_list_CC)
		q_list_sus_sig = q_list_signature(q_list_sus)
		FT_vecs_q_CC = zeros(ComplexF64, N, length(q_list_CC))
		FT_vecs_q_sus = zeros(ComplexF64, N, length(q_list_sus))
		fill_FT_table!(FT_vecs_q_CC, q_list_CC, x_array, y_array)
		fill_FT_table!(FT_vecs_q_sus, q_list_sus, x_array, y_array)

		# Path segments for spectral function - TRIANGULAR PATH:
		# (0,0) → (π,0) → (π,π) → (0,0)
		#
		# k-space convention: index div(L,2)+1 = k=0, index 1 = k=-π ≡ π
		# For both even and odd L: use index 1 for the π point
		
		# Key indices
		kx_pi_idx = 1  # π index for kx (always index 1)
		ky_pi_idx = 1  # π index for ky (always index 1)
		kx_0_idx = div(Lx_gen, 2) + 1  # 0 index for kx
		ky_0_idx = div(Ly_gen, 2) + 1  # 0 index for ky
		
		# Segment 1: (0,0) → (π,0) - kx from 0 towards π, ky stays at 0
		# Goes from kx_0_idx to Lx_gen, then wraps to 1
		seg1_kx_temp = div(Lx_gen,2)+1:div(Lx_gen,2)+(Lx_gen - div(Lx_gen, 2))
		seg1_kx = Int.([mod1(k, Lx_gen) for k in seg1_kx_temp])
		seg1_ky = Int.(fill(ky_0_idx, length(seg1_kx)))
		
		# Segment 2: (π,0) → (π,π) - kx stays at π (index 1), ky from 0 towards π
		seg2_len = Ly_gen - div(Ly_gen, 2)
		seg2_ky_temp = div(Ly_gen,2)+1:div(Ly_gen,2)+seg2_len
		seg2_ky = Int.([mod1(k, Ly_gen) for k in seg2_ky_temp])
		seg2_kx = Int.(fill(kx_pi_idx, length(seg2_ky)))
		
		# Segment 3: (π,π) → (0,0) - DIAGONAL back to origin
		# Both kx and ky go from π (index 1) back to 0 (index div+1) simultaneously
		seg3_len = Lx_gen - div(Lx_gen, 2)
		
		# kx goes from 1 (π) through Lx,Lx-1,...,div+1 (0)
		seg3_kx = Int.(vcat([1], Lx_gen:-1:kx_0_idx+1))
		seg3_kx = seg3_kx[1:seg3_len]  # Take only seg3_len points
		
		# ky goes from 1 (π) through Ly,Ly-1,...,div+1 (0)
		seg3_ky = Int.(vcat([1], Ly_gen:-1:ky_0_idx+1))
		# Interpolate ky to match seg3_len points (handle Lx ≠ Ly)
		if length(seg3_ky) != seg3_len
			# Linear interpolation of indices
			ky_indices_float = range(1, length(seg3_ky), length=seg3_len)
			seg3_ky = Int.([seg3_ky[round(Int, i)] for i in ky_indices_float])
		end
		
		# Add final (0,0) point - same as first point, closes the triangle
		kx_ix_path = Int.(vcat(seg1_kx, seg2_kx, seg3_kx, kx_0_idx))
		ky_ix_path = Int.(vcat(seg1_ky, seg2_ky, seg3_ky, ky_0_idx))
		k_path = [kx_ix_path, ky_ix_path]



		q_tuple_list = [(0.0, q) for q in (2*pi/Ly) * (0:(div(Ly,2)-1))]
		append!(q_tuple_list, [(q, 0.0) for q in (2*pi/Lx) * (1:(div(Lx,2)-1))]) # preallocate q_tuple_list
		q_ix_tuples = [(Int64(round(qx*Lx/(2*pi) + 1)), Int64(round(qy*Ly/(2*pi) + 1))) for (qx, qy) in q_tuple_list]


		kx_list = get_kx_list(Lx)
		ky_list = get_ky_list(Ly)
		k_vectors = [[kx, ky] for kx in kx_list for ky in ky_list]

		# General k-vectors (for rectangular, use Lx*Ly)
		k_general_vectors = [zeros(Float64, 2) for _ in 1:(N * SC_L^2)]
		k_general_idx = zeros(Int64, Lx*SC_L, Ly*SC_L)
		k_general_xy = [zeros(Int64, 2) for _ in 1:(N * SC_L^2)]
		k_general_qK_idx = [zeros(Int64, 2) for _ in 1:(N * SC_L^2)]
		
		CI_k = make_CartesianIndices(Lx, Ly)  # For k-space indexing
		for qix in 1:N
			q_vec = k_vectors[qix]
			qx = get_x(qix, CI_k)
			qy = get_y(qix, CI_k)
			for Kix in 1:SC_L^2
				K_vec = SC_K_vectors[Kix]
				Kx = get_x(Kix, SC_L)
				Ky = get_y(Kix, SC_L)
				kg_ix = idx((qx-1)*SC_L + Kx, (qy-1)*SC_L + Ky, Lx*SC_L)
				kg = [q_vec[1] + K_vec[1], q_vec[2] + K_vec[2]]

				k_general_vectors[kg_ix] = kg
				k_general_qK_idx[kg_ix] = [qix, Kix]
				kgx = get_x(kg_ix, Lx*SC_L)
				kgy = get_y(kg_ix, Lx*SC_L)
				k_general_xy[kg_ix] = [kgx, kgy]
				k_general_idx[kgx, kgy] = kg_ix
			end
		end		

		d_wave_pairings = ["d-wave", "d-p-wave", "mayr-d-wave", "d+s-wave"]
		s_wave_pairings = ["s-wave", "d+s-wave"]
		if sim_data.pairing_symmetry != "s-wave"
			@warn "pairing_symmetry=\"$(sim_data.pairing_symmetry)\": the d-wave / mixed pairing " *
				  "code is NOT VALIDATED. It was never benchmarked against published results, and " *
				  "the factor of the bond pairing term is not settled (2 in MCMF_action_K and " *
				  "MCMF_action_K0, 1 in MCMF_action_TCluster_K0). Only \"s-wave\" is production-ready." maxlog=1
		end

		# Density-channel modulation sign s_i = cos(Q.r_i). For the supported
		# ordering vectors (Q=0 and Q=(pi,pi)) this is exactly +-1; cos() keeps it
		# real so the BdG diagonal stays real (phi is a real field).
		Qd = sim_data.Q_density
		density_sign = [cos(Qd[1] * x_array[i] + Qd[2] * y_array[i]) for i in 1:N]


		return ConstStruct(Lx
				, Ly
				, N
				, CI
				, LI
				, x_array
				, y_array
				, dist
				, NN_mat
				, NN_x
				, NN_y
				, V_matrix
				, U_params
				, U_params_x
				, U_params_y
				, w_delta
				, kx_list
				, ky_list
				, k_vectors
				, spins
				, new_ix_list
				, new_iy_list
				, ls_list
				, ls_list_reduced
				, k_path
				, q_list_CC
				, q_list_sus
				, q_list_CC_sig
				, q_list_sus_sig
				, FT_vecs_q_CC
				, FT_vecs_q_sus
				, q_tuple_list
				, q_ix_tuples
				, Tcluster_NN_matrix
				, TCluster_assignments
				, SC_Ki_list
				, SC_K_vectors
				, SC_minus_K
				, k_general_vectors
				, k_general_xy
				, k_general_idx
				, k_general_qK_idx
				, xedges
				, d_wave_pairings
				, s_wave_pairings
				, pierls_phases
				, density_sign
				, mu_shift
				)
	end

	"""
	    make_V_matrix(Lx, Ly, V; seed=nothing) -> Vector{Float64}
	    make_V_matrix(L, V; seed=nothing)

	On-site potential disorder V_i ~ Uniform[-V/2, V/2] (enters the Hamiltonian as μ - V_i).
	With `seed` it uses its own random generator (seed + 2), so the map depends only on
	the disorder seed.
	"""
	function make_V_matrix(Lx::Int64, Ly::Int64, V::Float64; seed=nothing)
		# Uses a LOCAL MersenneTwister rather than reseeding the global RNG: the
		# realization must depend only on its own seed, not on whatever the
		# global stream happened to be doing. (This changes nothing for existing
		# data — every run so far has V=0, so V_matrix is zeros either way, and
		# get_random_seed!() reseeds the global RNG immediately afterwards.)
		N = Lx * Ly
		rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed + 2)
		V_matrix = zeros(Float64, N)
		for i in 1:N
			V_matrix[i] = (rand(rng) - 0.5) * V
		end
		return V_matrix
	end

	# Legacy wrapper for square lattice
	make_V_matrix(L::Int64, V::Float64; seed=nothing) = make_V_matrix(L, L, V; seed=seed)

	"""
	    build_mu_disorder!(U_params, sim_data; seed=nothing) -> mu_shift

	Binary μ–U disorder (`sim_data.mu_disorder`, fraction f = `mu_disorder_f`):
	exactly ⌊fN⌋ random sites are "empty" (U_i = 0, μ unshifted), the rest "full".
	- `"interaction-only"`: U_i ∈ {0, U}, μ uniform;
	- `"coupled"`: U_i ∈ {0, U} and μ_i = μ - U/2 on the full sites;
	- `"potential-only"`: U stays uniform, only μ is shifted on the full sites;
	- `"none"`: nothing.
	All modes use the same partition for a given (f, seed), so they can be compared.
	Overwrites `U_params` for the first two modes (s-wave only) and returns the per-site
	shift U_i/2 (to be added to `V_matrix`).
	"""
	function build_mu_disorder!(U_params::Array{Float64,1}, sim_data::SimData; seed=nothing)
		# Binary mu-U disorder with U_i in {0, U}. Three modes (isolation trio):
		#   "interaction-only" : U_i in {0,U} (interaction on/off), mu NOT shifted.
		#   "coupled"          : U_i in {0,U} AND mu_i = mu - U_i/2.
		#   "potential-only"   : interaction stays uniform U, only mu is shifted
		#                        (mu -> mu - U/2 on the (1-f) sites).
		#
		# All three draw the SAME EXACT partition of floor(f*N) "empty" sites from the
		# DISORDER seed (local RNG, reproducible, does not disturb the global stream --
		# same convention as make_V_matrix). So at fixed (f, disorder_seed) the three
		# modes share one site partition and differ only in whether U and/or mu is
		# touched -- exactly what isolates interaction- vs potential-disorder effects.
		# Returns the per-site chemical-potential shift mu_shift (= U_i/2, or 0).
		N = sim_data.N
		mu_shift = zeros(Float64, N)
		mode = sim_data.mu_disorder
		mode == "none" && return mu_shift
		if !(mode in ("coupled", "potential-only", "interaction-only"))
			error("Unknown mu_disorder: \"$(mode)\" (expected \"none\", \"coupled\", " *
				  "\"potential-only\", or \"interaction-only\")")
		end
		f = sim_data.mu_disorder_f
		(0.0 <= f <= 1.0) || error("mu_disorder_f must be in [0,1], got $f")
		U = sim_data.U

		n_zero = floor(Int, f * N)                 # EXACT count of U_i = 0 sites
		rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed + 3)
		perm = Random.randperm(rng, N)
		zero_sites  = perm[1:n_zero]               # U_i = 0  (interaction off / mu unshifted)
		shift_sites = perm[n_zero+1:end]           # U_i = U  (interaction on  / mu shifted)

		# mu shift = U_i/2 on the interacting sites -- EXCEPT interaction-only, which
		# disorders the interaction but leaves the chemical potential uniform.
		if mode != "interaction-only"
			mu_shift[shift_sites] .= U / 2
		end

		# Genuinely disordered interaction (U on the shifted sites, 0 elsewhere) for the
		# two modes that turn the interaction on/off. potential-only keeps U uniform.
		# (On-site s-wave U only; d-wave bond U is not covered.)
		if mode in ("coupled", "interaction-only")
			sim_data.pairing_symmetry == "s-wave" ||
				error("mu_disorder=\"$(mode)\" is only implemented for s-wave pairing")
			fill!(U_params, U)
			U_params[zero_sites] .= 0.0
		end
		return mu_shift
	end

	"""
	    make_U_params(U, Lx, Ly, U_disorder; impurity=1.0, U_puddle_params=(0.0, 10.0, 0.7, 0.7),
	                  seed=nothing, signs=[1], smooth=false, smooth_sigma=0.8,
	                  pairing_symmetry="s-wave", assert_positive=true, assert_mean=nothing,
	                  min_cutoff=0.0) -> (U_site, U_x, U_y)

	Build the site-dependent attraction U_i (length `Lx*Ly`) for the disorder model `U_disorder`.

	`impurity` sets the disorder strength; what it means depends on the model:

	| `U_disorder` | U_i | `impurity` | `U_puddle_params` used |
	|---|---|---|---|
	| `"uniform"` | U on every site | – | – |
	| `"rand"` | uniform random in [0, U] | – | – |
	| `"ON-OFF"` | U on a random 30% of the sites, 0 elsewhere | – | – |
	| `"puddles"` | V0 + Σ_j s_j VI e^(-R/λ)/R, R = √(r_ij² + rz²) | fraction of impurity sites | (V0, VI, λ, rz) |
	| `"gaussian"` | drawn from Normal(U, `impurity`), optionally smoothed | width of the distribution | – |
	| `"impurity-gaussian-kernel"` | (U/𝒩) Σ_j exp(-r_ij²/2σ²) | fraction p of interaction centers | σ = `U_puddle_params[3]` |
	| `"on-off-puddles"` | U inside disks of radius ceil(λ), 0 outside | fraction of the lattice covered | λ = `U_puddle_params[3]` |

	- `"impurity-gaussian-kernel"` is the model of the thesis (Ch. 6): 𝒩 is chosen so that
	  p = 1 gives U_i = U, and the mean is Ū = pU (it is not rescaled to U).
	- `"puddles"` is the pairing model of Sulangi et al., "Inhomogeneity, Fluctuations, and
	  Gap Filling in Overdoped Cuprates" (2025): a weak background V0 plus a Yukawa bump
	  around each impurity. `signs` gives the possible signs s_j of the bumps; terms
	  ≤ `min_cutoff` are dropped.
	- `smooth`/`smooth_sigma` (the `smooth_U` parameter of a run) apply only to `"gaussian"`.
	- `seed` (the disorder seed of the run) fixes the random map; `nothing` uses the
	  current global random state.
	- `assert_positive`: negative U_i are set to 0, so no site becomes repulsive.
	- `assert_mean`: rescale the map so its mean is U. Default: on for every model
	  except `"impurity-gaussian-kernel"`.

	Returns the on-site couplings and the bond couplings in x and y (the bond value is
	the average of the two sites). Which of the three are non-zero depends on
	`pairing_symmetry`: s-wave uses only the on-site values, d-wave and extended-s only
	the bond values, d+s-wave both (the non-s-wave cases are experimental, see README).
	"""
	function make_U_params(U::Float64, Lx::Int64, Ly::Int64, U_disorder::String; U_puddle_params=(0.0, 10.0, 0.7, 0.7), impurity=1.0, seed=nothing, signs=[1], smooth=false, smooth_sigma=0.8, pairing_symmetry="s-wave", assert_positive=true, assert_mean=nothing, min_cutoff=0.0)
		N = Lx * Ly
		if U_disorder == "uniform"
			U_params = ones(N) * U
		elseif U_disorder == "rand"
			if seed !== nothing
				Random.seed!(seed)
			end
			U_params = rand(N) * U
		elseif U_disorder == "ON-OFF"
			if seed !== nothing
				Random.seed!(seed)
			end
			U_params = zeros(N)
			for i in 1:N
				if rand() < 0.3
					U_params[i] = U
				else
					U_params[i] = 0.0
				end
			end
		elseif U_disorder == "puddles"
			V0, VI, lambda_U, rz = U_puddle_params
			U_params = make_U_puddles(Lx, Ly; V0=V0, VI=VI, λ=lambda_U, rz=rz, impurity_ratio=impurity, seed=seed, signs=signs, min_cutoff=min_cutoff)
		elseif U_disorder == "gaussian"
			# take U_puddle_params[5] (impurity ratio) as the std of the gaussian
			U_params = make_U_gaussian(Lx, Ly, U, impurity; seed=seed, smooth=smooth, smooth_sigma=smooth_sigma)
		elseif U_disorder == "impurity-gaussian-kernel"
			# impurity is the occupied impurity fraction p, and U_puddle_params[3] stores the Gaussian width
			sigma = U_puddle_params[3]
			U_params = make_U_impurity_gaussian_kernel(Lx, Ly, U; impurity_fraction=impurity, sigma=sigma, seed=seed)
		elseif U_disorder == "on-off-puddles"
			# impurity is the covering precent here
			# lambd = U_puddle_params[3] is the puddle size
			# U is the value inside the puddles
			puddle_size = U_puddle_params[3]
			num_puddles = round(Int, impurity * N / (puddle_size^2 ))
			U_params = make_U_on_off_puddles(Lx, Ly, U; num_puddles=num_puddles, puddle_size=puddle_size, seed=seed)
		else
			error("Unknown U disorder type: $U_disorder")
		end

		if assert_positive
			U_params[findall(U_params.<0.0)] .= 0.0
		end

		# For impurity-gaussian-kernel, keep natural mean(U) vs p unless explicitly requested.
		resolved_assert_mean = if assert_mean === nothing
			U_disorder != "impurity-gaussian-kernel"
		else
			Bool(assert_mean)
		end

		# assert the mean is U
		if resolved_assert_mean && mean(U_params) != 0.0
			U_params .*= U / mean(U_params)
		end

		U_x = zeros(Float64, N)
		U_y = zeros(Float64, N)
		for i in 1:N
			xi = get_x(i, Lx)
			yi = get_y(i, Lx)
			U_x[i] = (U_params[i] + U_params[idx(add_xy_ix(xi, 1, Lx), yi, Lx)])/ 2
			U_y[i] = (U_params[i] + U_params[idx(xi, add_xy_ix(yi, 1, Ly), Lx)])/ 2
		end

		if pairing_symmetry == "s-wave"
			return U_params, zeros(Float64, N), zeros(Float64, N)
		elseif pairing_symmetry in ["d-wave", "d-p-wave", "mayr-d-wave"]
			return zeros(Float64, N), U_x, U_y
		elseif pairing_symmetry == "extended-s"
			return zeros(Float64, N), U_x, U_x
		elseif pairing_symmetry == "d+s-wave"
			return U_params, U_x, U_y
		else
			error("Unknown pairing symmetry: $pairing_symmetry")
		end
	end

	# Legacy wrapper for square lattice
	make_U_params(U::Float64, L::Int64, U_disorder::String; kwargs...) = make_U_params(U, L, L, U_disorder; kwargs...)


	"""
	    make_U_gaussian(Lx, Ly, dist_mean, dist_std; seed=nothing, smooth=false, smooth_sigma=0.8)

	Draw U_i independently from Normal(`dist_mean`, `dist_std`) on every site.
	With `smooth`, the map is then blurred with a Gaussian filter of width `smooth_sigma`
	(lattice units). No clipping or rescaling here; `make_U_params` does that.
	"""
	function make_U_gaussian(Lx::Int64, Ly::Int64, dist_mean::Float64, dist_std::Float64; seed=nothing, smooth=false, smooth_sigma=0.8)
		N = Lx * Ly
		if seed !== nothing
			Random.seed!(seed)
		end

		U_distribution = Normal(dist_mean, dist_std)
		U_params = rand(U_distribution, N)
		if smooth
			# smooth the disorder a bit
			U_params_mat = reshape(U_params, Lx, Ly)
			U_params_mat = imfilter(U_params_mat, Kernel.gaussian(smooth_sigma))
			U_params = vec(U_params_mat)
		end
		return U_params
	end

	# Legacy wrapper for square lattice
	make_U_gaussian(L::Int64, dist_mean::Float64, dist_std::Float64; kwargs...) = make_U_gaussian(L, L, dist_mean, dist_std; kwargs...)


	"""
	    make_U_impurity_gaussian_kernel(Lx, Ly, U; impurity_fraction=1.0, sigma=1.0, seed=nothing)

	Interaction-center disorder of the thesis (Ch. 6, eq. U_disorder_landscape):
	U_i = (U/𝒩) Σ_j exp(-|r_i - r_j|² / 2σ²), summed over round(p·N) random centers j,
	p = `impurity_fraction`.

	Distances use the nearest periodic image. 𝒩 is the sum of the kernel over the whole
	periodic lattice, so p = 1 gives exactly U_i = U and the mean is U·p.
	Returns all zeros if there are no centers.
	"""
	function make_U_impurity_gaussian_kernel(Lx::Int64, Ly::Int64, U::Float64; impurity_fraction::Float64=1.0, sigma::Float64=1.0, seed=nothing)
		N = Lx * Ly
		if seed !== nothing
			Random.seed!(seed)
		end

		sigma > 0.0 || error("sigma must be positive in make_U_impurity_gaussian_kernel")
		p = clamp(impurity_fraction, 0.0, 1.0)
		num_impurities = Int(round(p * N))

		if num_impurities == 0
			return zeros(Float64, N)
		end

		impurity_indices = randperm(N)[1:num_impurities]
		sigma2_2 = 2.0 * sigma^2

		# Normalize on the periodic lattice so Σ_{Δr} K(Δr) = 1.
		kernel_norm = 0.0
		for dx in 0:(Lx - 1)
			dx_wrap = dx > Lx/2 ? dx - Lx : dx
			for dy in 0:(Ly - 1)
				dy_wrap = dy > Ly/2 ? dy - Ly : dy
				r2 = dx_wrap^2 + dy_wrap^2
				kernel_norm += exp(-r2 / sigma2_2)
			end
		end

		inv_norm = 1.0 / kernel_norm
		U_params = zeros(Float64, N)
		for i in 1:N
			xi = get_x(i, Lx)
			yi = get_y(i, Lx)
			ui = 0.0
			for j in impurity_indices
				xj = get_x(j, Lx)
				yj = get_y(j, Lx)
				dx = abs(xi - xj)
				dy = abs(yi - yj)
				# shortest image convention for PBC
				if dx > Lx/2
					dx -= Lx
				end
				if dy > Ly/2
					dy -= Ly
				end
				r2 = dx^2 + dy^2
				ui += exp(-r2 / sigma2_2)
			end
			U_params[i] = U * ui * inv_norm
		end

		return U_params
	end

	# Legacy wrapper for square lattice
	make_U_impurity_gaussian_kernel(L::Int64, U::Float64; kwargs...) = make_U_impurity_gaussian_kernel(L, L, U; kwargs...)


	"""
	    make_U_puddles(Lx, Ly; impurity_ratio=0.075, V0=0.8, VI=3.125, rz=0.7, λ=0.7,
	                   signs=[1], seed=nothing, min_cutoff=0.0)

	Pairing landscape of Sulangi et al. (2025): a homogeneous background `V0` plus a
	Yukawa bump s_j·VI·e^(-R/λ)/R around each impurity, R = √(r² + rz²).

	round(`impurity_ratio`·N) impurity sites are chosen at random; each bump's sign s_j is
	drawn from `signs`. `rz` keeps R > 0 on the impurity site itself. A bump value
	≤ `min_cutoff` at a site is dropped. Distances use the nearest periodic image.
	"""
	function make_U_puddles(Lx::Int64, Ly::Int64; impurity_ratio=0.075, V0=0.8, VI=3.125, rz=0.7, λ=0.7, signs=[1], seed=nothing, min_cutoff=0.0)
		N = Lx * Ly
		if seed !== nothing
			Random.seed!(seed)
		end
		U_params = ones(Float64, N) * V0
		num_impurities = Int(round(impurity_ratio * N))
		impurity_indices = randperm(N)[1:num_impurities]
		impurity_signs = rand(signs, num_impurities)
		for i in 1:N
			xi = get_x(i, Lx)
			yi = get_y(i, Lx)
			for (impurity_ix ,j) in enumerate(impurity_indices)
				xj = get_x(j, Lx)
				yj = get_y(j, Lx)
				dx = abs(xi - xj)
				dy = abs(yi - yj)
				# shortest image convention for PBC
				if dx > Lx/2
					dx -= Lx
				end
				if dy > Ly/2
					dy -= Ly
				end
				rxy2 = dx^2 + dy^2
				R = sqrt(rxy2 + rz^2)
				imp_add = impurity_signs[impurity_ix] * VI * exp(-R / λ) / R
				imp_add = imp_add <= min_cutoff ? 0.0 : imp_add
				U_params[i] += imp_add
			end
		end
		return U_params
	end

	# Legacy wrapper for square lattice
	make_U_puddles(L::Int64; kwargs...) = make_U_puddles(L, L; kwargs...)

	"""
	    make_U_on_off_puddles(Lx, Ly, U; num_puddles=1, puddle_size=1.0, seed=nothing)

	Put `num_puddles` disks of radius ceil(`puddle_size`) at distinct random centers;
	a site gets U for every disk it lies in (overlapping disks add), 0 outside all disks.
	"""
	function make_U_on_off_puddles(Lx::Int64, Ly::Int64, U::Float64; num_puddles::Int64=1, puddle_size::Float64=1.0,  seed=nothing)
		N = Lx * Ly
		if seed !== nothing
			Random.seed!(seed)
		end
		U_params = zeros(Float64, N)
		puddles_ix_list = rand(1:N, num_puddles)
		puddle_size = ceil(puddle_size)
		i = 0
		while length(unique(puddles_ix_list)) < num_puddles
			puddles_ix_list = unique(puddles_ix_list)
			new_ix = rand(1:N, num_puddles - length(puddles_ix_list))
			append!(puddles_ix_list, new_ix)
			i += 1
			if i > N
				break
			end
		end
		for i in 1:N
			xi = get_x(i, Lx)
			yi = get_y(i, Lx)
			for j in puddles_ix_list
				xj = get_x(j, Lx)
				yj = get_y(j, Lx)
				dx = abs(xi - xj)
				dy = abs(yi - yj)
				# shortest image convention for PBC
				if dx > Lx/2
					dx -= Lx
				end
				if dy > Ly/2
					dy -= Ly
				end
				R = sqrt(dx^2 + dy^2)
				if R < puddle_size
					U_params[i] += U
				end
			end
		end


		return U_params
	end

	# Legacy wrapper for square lattice
	make_U_on_off_puddles(L::Int64, U::Float64; kwargs...) = make_U_on_off_puddles(L, L, U; kwargs...)

	"""
	    get_random_seed!() -> Int

	Reseed the global random generator from the clock and return the seed. Called after
	the disorder maps are drawn, so the MC stream does not repeat between runs.
	"""
	function get_random_seed!()
		seed = Int(Dates.now().instant.periods.value % Int(1e9))
		Random.seed!(seed)
		return seed
	end

	"""
	    minus_K_ix(ix, L_SC) -> Int

	Index of -K for the supercell wavevector with index `ix` (supercell method).
	"""
	function minus_K_ix(ix::Int64, L_SC::Int64)
		kx = mod(ix - 1, L_SC)
		ky = Int((ix - kx - 1)/L_SC)
		kx_new = mod(-kx, L_SC)
		ky_new = mod(-ky, L_SC)
		return kx_new + ky_new * L_SC + 1
	end

	"""
	    assert_flux_quantization!(sim_data; print_B_change=true)

	With periodic boundaries, round `sim_data.B` (in place) to the nearest allowed value:
	B·Ly (`"LandauX"`), B·Lx (`"LandauY"`) or B·Lx·Ly (`"Mixed"`) must be a multiple of 2π.
	Prints the change unless `print_B_change=false`. Does nothing for B = 0.
	"""
	function assert_flux_quantization!(sim_data::SimData; print_B_change=true)
		# L depends on gauge choice:
		#   LandauX: A=(-By,0) -> phase along x -> need B*Ly = 2πn
		#   LandauY: A=(0,Bx)  -> phase along y -> need B*Lx = 2πn
		#   Mixed:   A=(B*y*Lx,B*x*Ly) -> phases along both x and y -> need B*Lx*Ly = 2πn

		# The flux over a plaquate is B, so the total flux through the system is B*Lx*Ly.
		
		if sim_data.boundary_conditions != "PBC" || sim_data.B == 0.0
			return
		end
		
		# Determine which length L to use based on gauge choice
		if sim_data.gauge_choice == "LandauX"
			L = sim_data.Ly
			L_name = "Ly"
		elseif sim_data.gauge_choice == "LandauY"
			L = sim_data.Lx
			L_name = "Lx"
		elseif sim_data.gauge_choice == "Mixed"
			L = sim_data.Lx * sim_data.Ly
			L_name = "Lx*Ly"
		else
			error("Gauge choice $(sim_data.gauge_choice) not recognized.")
		end
		
		# Check flux quantization condition
		if sim_data.B * L % (2*pi) != 0.0
			old_B = sim_data.B
			sim_data.B = 2*pi * round(Int, sim_data.B * L / (2*pi)) / L
			if print_B_change
				println("Note: For PBC with $(sim_data.gauge_choice) gauge, B*$(L_name) must be an integer multiple of 2π.")
				println("Given B=$(old_B), $(L_name)=$(L), B*$(L_name)/2π=$(old_B*L/(2*pi)).")
				println("Adjusting B to the nearest value that satisfies this condition.")
				println("New B=$(sim_data.B).")
			end
		end
	end

	"""
	    make_peierls_phases(sim_data) -> Matrix{ComplexF64}  (N × 2)

	Peierls phase of the +x (column 1) and +y (column 2) hop from each site, for the
	gauges of `neighbor_matrix`. Enforces flux quantization first
	(`assert_flux_quantization!`, which may change `sim_data.B`). All ones for B = 0.
	"""
	function make_peierls_phases(sim_data::SimData)
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		B = sim_data.B
		gauge_choice = sim_data.gauge_choice
		peierls_phases = ones(ComplexF64, N, 2) # peierls phases for each site and direction (x,y)
		# for B=0 there are no phases
		if B==0.0
			return peierls_phases
		end
		# Enforce flux quantization for PBC
		assert_flux_quantization!(sim_data)
		# set phases
		for ix in 1:N
			xi = get_x(ix, Lx)
			yi = get_y(ix, Lx)
			if gauge_choice == "LandauX"
				# phase for hopping in x direction
				phase_x = exp(-1im * sim_data.B * yi)
				peierls_phases[ix, 1] = phase_x
				# no phase for hopping in y direction in this gauge
				peierls_phases[ix, 2] = 1.0
			elseif gauge_choice == "LandauY"
				# phase for hopping in y direction
				phase_y = exp(1im * sim_data.B * xi)
				peierls_phases[ix, 2] = phase_y
				# phase for hopping in x direction is 1 in landau gauge
				peierls_phases[ix, 1] = 1.0
			elseif gauge_choice == "Mixed"
				# phase for hopping in y direction
				phase_y = exp(1im * sim_data.B * xi * Ly)
				peierls_phases[ix, 2] = phase_y
				# phase for hopping in x direction
				phase_x = exp(1im * sim_data.B * yi * Lx)
				peierls_phases[ix, 1] = phase_x
			else
				error("Gauge choice $(gauge_choice) not recognized.")
			end

		end
		return peierls_phases
	end

	"""
	    Temporary_Data(sim_data) -> TemporaryData

	Allocate the scratch space (see `TemporaryData`). Large measurement buffers are left
	empty here and allocated on the first full measurement.
	"""
	function Temporary_Data(sim_data::SimData)
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		L_SC = sim_data.SC_L
		N_SC = L_SC^2
		G_length = maximum([max(Lx, Ly) + 1, N_SC])
		S_r = zeros(ComplexF64, 0, 0)
		plan_S_r_placeholder = plan_fft(zeros(ComplexF64, 1, 1))
		# TCluster dimensions
		TCluster_N = sim_data.TCluster_size_x * sim_data.TCluster_size_y
		len_CC_q = 1 + div(Lx,2) + div(Ly,2)  # q=0 plus finite-q current-current vectors
		maxEoverN = 4.0/N # maximum E/N expected, used for scaling delta in analytic continuation
		out = TemporaryData(
							zeros(Float64, 2*N, N_SC), # Energies_Bckup
							zeros(ComplexF64, 2*N, 2*N, N_SC), # U_matrix_Bckup
							zeros(ComplexF64, 2*N, 2*N), # K matrix
							zeros(Float64, 2*N), # Energies_K
							zeros(ComplexF64, 2*N, 2*N), # U_matrix_K
							zeros(Float64, 2), # Delta_suggestion
							HermitianEigenWs(zeros(ComplexF64, 2*N, 2*N), vecs=true), # workspace for diagonalization
							zeros(ComplexF64, N*N_SC), # spectral function
							zeros(Float64, 2*N, N_SC), # FD distribution
							zeros(Float64, 2*N, N_SC), # exp(beta*E)
							maxEoverN, # delta (initialized to maxEoverN, will be updated based on energy spacings)
							maxEoverN, # delta_max (initialized to maxEoverN)
							zeros(ComplexF64, 0, 0), # M matrix - dead
							zeros(ComplexF64, 0), # v_i (legacy placeholder)
							S_r, # S_r
							zeros(ComplexF64, N), # S_r_vec
							zeros(ComplexF64, N), # G_r_vec
							zeros(ComplexF64, N, G_length), # FT_vecs_q
							zeros(ComplexF64, G_length), # G_q_proj
							zeros(ComplexF64, 0, 0), # S_q (legacy placeholder)
							zeros(ComplexF64, 0, 0), # G_matrix (legacy placeholder)
							zeros(ComplexF64, 2*N), #g_A
							zeros(ComplexF64, 2*N), # g_B
							plan_S_r_placeholder, # plan_Sr (legacy placeholder)
							Array{ComplexF64,2}[], # S_up_x (legacy path, lazy)
							Array{ComplexF64,2}[], # S_down_x (legacy path, lazy)
							Array{ComplexF64,2}[], # S_up_y (legacy path, lazy)
							Array{ComplexF64,2}[], # S_down_y (legacy path, lazy)
							Array{ComplexF64,2}[], # G_matrix_vec_x (lazy: ensure_G_buffers!, G_length matrices)
							Array{ComplexF64,2}[], # G_matrix_vec_y (lazy: ensure_G_buffers!)
							zeros(Float64, N), #LDOS
							zeros(ComplexF64, 2*TCluster_N, 2*TCluster_N), # TCluster_K
							zeros(Float64, TCluster_N), # TCluster_V
							zeros(ComplexF64, TCluster_N), # TCluster_Delta_s
							zeros(ComplexF64, TCluster_N), # TCluster_Delta_dx
							zeros(ComplexF64, TCluster_N), # TCluster_Delta_dy
							zeros(Float64, TCluster_N), # TCluster_phi
							zeros(Float64, TCluster_N), # TCluster_dsign
							HermitianEigenWs(zeros(ComplexF64, 2*TCluster_N, 2*TCluster_N), vecs=false), # TCluster_ws (eigenvalues-only for TCA)
							zeros(Float64, 2*TCluster_N), # TCluster_energies
							zeros(ComplexF64, 2*TCluster_N, 2*TCluster_N), # TCluster_U_matrix
							zeros(ComplexF64, len_CC_q), # current_operator_q
							zeros(ComplexF64, 0, 0), # sus_weighted_up (lazy: sized by susceptibility_q_vec_both!)
							zeros(ComplexF64, 0, 0), # sus_weighted_dn_conj (lazy)
							nothing # meas_work (lazy: measurement_work!)
							)
		return out
	end

	"""
	    invalidate_global_diag!(data)

	Mark the stored full-lattice eigen-solution (S, Energies, U_matrix) as out of date,
	e.g. after a K0-TCA update that changed a field without a full diagonalization.
	"""
	function invalidate_global_diag!(data::LatticeData)
		data.current_state.diag_is_synced = false
		return nothing
	end

	"""
	    mark_global_diag_synced!(data)

	Mark the stored full-lattice eigen-solution as matching the current fields.
	"""
	function mark_global_diag_synced!(data::LatticeData)
		data.current_state.diag_is_synced = true
		data.current_state.diag_synced_step = data.current_state.step_count
		return nothing
	end

	"""
	    ensure_global_diag_synced!(data; reason="") -> Bool

	If the stored eigen-solution is out of date, diagonalize the full lattice and store
	S, Energies and U_matrix. Returns true if it had to. Called before every measurement.
	"""
	function ensure_global_diag_synced!(data::LatticeData; reason::String="")
		if data.current_state.diag_is_synced
			return false
		end

		temp_S, temp_energies, temp_U_matrix = MCMF_action_K(data, data.current_state.Delta_s, data.current_state.Delta_dx, data.current_state.Delta_dy)
		data.current_state.S = temp_S
		data.current_state.Energies .= temp_energies
		data.current_state.U_matrix .= temp_U_matrix
		mark_global_diag_synced!(data)
		return true
	end


	"""
	    hist_bin_clamped(x, lo, hi, nbins) -> Int

	Bin of `x` among `nbins` equal bins over [lo, hi]. Values outside the range go into
	the first or last bin, so those bins also count everything below or above the range.
	"""
	@inline function hist_bin_clamped(x::Float64, lo::Float64, hi::Float64, nbins::Int64)
		t =(x - lo) / (hi - lo)
		b = floor(Int, t * nbins) + 1
		return clamp(b, 1, nbins)
	end

	"""
	    ensure_field_histograms!(data)

	Make the φ and Δ histogram accumulators the size `sim_data` asks for. They are
	reallocated only if the size is wrong (old checkpoints without histograms, changed
	bin counts), so a normal resume keeps its counts. 0 bins = histogram off (empty).
	"""
	function ensure_field_histograms!(data::LatticeData)
		cs =data.current_state
		sd = data.sim_data

		# phi: 1D
		nphi = max(0, sd.phi_hist_nbins)
		if length(cs.phi_hist) != nphi
			if !isempty(cs.phi_hist) && nphi > 0
				@warn "Resizing phi histogram; accumulated counts discarded" old_len=length(cs.phi_hist) new_len=nphi
			end
			cs.phi_hist = zeros(Int64, nphi)
		end

		# Delta: 2D (|Delta| x arg); ON only when BOTH axes have > 0 bins.
		na = max(0, sd.Delta_hist_abs_nbins)
		np = max(0, sd.Delta_hist_arg_nbins)
		if na == 0 || np == 0
			na = 0
			np = 0
		end
		if size(cs.Delta_hist) != (na, np)
			if !isempty(cs.Delta_hist) && na > 0 && np > 0
				@warn "Resizing Delta histogram; accumulated counts discarded" old_size=size(cs.Delta_hist) new_size=(na, np)
			end
			cs.Delta_hist = zeros(Int64, na, np)
		end
		return nothing
	end

	"""
	    make_measurement_K!(data, i; fast=true)

	Take one full measurement and store it at buffer index `i` (the last index if
	`i > saving_interval`; A, BZ and LDOS are updated as running means).

	First makes sure the eigen-solution matches the fields (`ensure_global_diag_synced!`),
	then computes the Fermi factors and the broadening δ, and measures: action, level
	spacing, n, spectral function / DOS / LDOS / BZ, kinetic energies Kx, Ky,
	current-current correlations Λab(q), the paramagnetic current (if `compute_jP`),
	fixed-δ and DC conductivities, L_D, pairing observables (Δ operator, DdD, Ps(l),
	phase correlations), susceptibilities, the φ observables and the field histograms.

	`fast=true` uses the fast kernels (same numbers up to summation order; `SC_L = 1`
	only), `fast=false` the reference kernels.
	"""
	function make_measurement_K!(data::LatticeData, i::Int64; fast::Bool=true)
		if i >data.sim_data.saving_interval
			i = data.sim_data.saving_interval
		end

		# Ensure observables use a state synchronized with current Delta fields.
		ensure_global_diag_synced!(data; reason="measurement")

		# preperations
		get_FD_dist!(data)
		# compute the delta parameter for Lorentzian broadening
		compute_delta_analitic_continuation!(data; method="mean_diff_low")

		measurements = data.measurement_data
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		N = data.sim_data.N
		SC_L = data.sim_data.SC_L
		L_gen_x = Lx * SC_L
		L_gen_y = Ly * SC_L
		N_gen = L_gen_x * L_gen_y

		# measurement buffers (allocated on the first measurement, then reused)
		ensure_G_buffers!(data)
		use_fast = fast && SC_L == 1
		work = use_fast ? measurement_work!(data) : nothing

	   
		# MCMF action
		measurements.Ss[i] = data.current_state.S

		# Energy spacing observables
		# Compute mean of all energy spacings and store delta_mean_diff_low
		Energies = data.current_state.Energies
		sorted_E = sort(vec(Energies))
		diffs = diff(sorted_E)
		if length(diffs) > 0
			measurements.energy_spacing_mean[i] = mean(diffs)
		else
			measurements.energy_spacing_mean[i] = 0.0
		end
		# delta_mean_diff_low is already computed and stored in temporary_data.delta
		measurements.delta_mean_diff_low[i] = data.temporary_data.delta

		# avrange occupation - n
		measurements.n[i] = avrage_n_K(data)

		# spectral function and DOS
		if use_fast
			measure_spectral_fast!(data, measurements, i, work)
		else
			ki_list = data.const_struct.ki_list
			kx_gen_ix_path, ky_gen_ix_path = data.const_struct.SF_k_path
			for (j,w) in enumerate(data.sim_data.w_list)
				#LDOS
				LDOS_c = compute_full_LDOS(data, w; delt = data.temporary_data.delta_max)
				measurements.LDOS[j] .*= i-1
				measurements.LDOS[j] .+= LDOS_c
				measurements.LDOS[j] ./= i

				# DOS
				DOS_c = sum(LDOS_c) / N_gen
				measurements.DOS[i][j] += DOS_c


				# Spectral function
				for l in eachindex(kx_gen_ix_path)
					kx_gen_ix = kx_gen_ix_path[l]
					ky_gen_ix = ky_gen_ix_path[l]
					k_gen_idx = data.const_struct.k_general_idx[kx_gen_ix, ky_gen_ix]
					q_ix, K_ix = data.const_struct.k_general_qK_idx[k_gen_idx]
					measurements.A[j][l] *= i-1
					measurements.A[j][l] += spectral_functionV2(data,K_ix, q_ix, w; delt=data.temporary_data.delta_max)
					measurements.A[j][l] /= i
				end

				if w == 0.0
					# BZ - Brillouin zone
					measurements.BZ .*= i-1
					measurements.BZ .+= spectral_function_fullV2(data, 0.0; delt=data.temporary_data.delta_max)
					measurements.BZ ./= i
				end
			end
		end

		## Kinetic energy in x direction
		measurements.Kx[i] = compute_Kx_operator(data; direction="x")
		## Kinetic energy in y direction
		measurements.Ky[i] = compute_Kx_operator(data; direction="y")


		# Measure all 4 current-current correlation components (Lxx, Lxy, Lyx, Lyy) at different q values
		if use_fast
			G_x_vec, G_y_vec = measure_current_correlations_q_fast!(data, measurements, i, work)
		else
			G_x_vec, G_y_vec = measure_current_correlations_q!(data, measurements, i; return_Gs=true)
		end
		SC_N = SC_L^2

		## FT of unperturbed paramagnetic current density <j^P_a(q)>_0 on the q_list_CC grid
		# Layout matches Lxx_q*: index 1 -> q=(0,0); next n_qy -> (0, qy); last n_qx -> (qx, 0)
		if data.sim_data.compute_jP
			n_qy = div(L_gen_y, 2)
			n_qx = div(L_gen_x, 2)
			qy_start = 2
			qy_end = 1 + n_qy
			qx_start = qy_end + 1
			qx_end = qy_end + n_qx
			jPx_vec = compute_jP_q(data; direction="x")
			jPy_vec = compute_jP_q(data; direction="y")
			measurements.jPx_q0[i] = jPx_vec[1]
			measurements.jPx_qy[i] .= jPx_vec[qy_start:qy_end]
			measurements.jPx_qx[i] .= jPx_vec[qx_start:qx_end]
			measurements.jPy_q0[i] = jPy_vec[1]
			measurements.jPy_qy[i] .= jPy_vec[qy_start:qy_end]
			measurements.jPy_qx[i] .= jPy_vec[qx_start:qx_end]
		end

		# Conductivity measurements
		# Reuse q=0 G matrices from q-correlation path when possible (SC_N==1).
		# For SC_N>1, keep the original K-resolved construction.
		if SC_N == 1
			data.temporary_data.G_matrix_vec_x[1] .= G_x_vec[1]
			data.temporary_data.G_matrix_vec_y[1] .= G_y_vec[1]
		else
			data.temporary_data.G_matrix_vec_x .= make_G_matrix_K0(data; K_list=[1:SC_N;], direction="x")[1]
			data.temporary_data.G_matrix_vec_y .= make_G_matrix_K0(data; K_list=[1:SC_N;], direction="y")[1]
		end

		# Fixed-delta conductivity measurements (using constant delta values)
		if use_fast
			# must run after L_ab(q): its packed pairs share memory with work.Mlm
			measure_fixed_delta_fast!(data, measurements, i, work, data.temporary_data.G_matrix_vec_x[1], data.temporary_data.G_matrix_vec_y[1])
		else
			for (d_idx, δ_fixed) in enumerate(data.sim_data.delta_values_fixed)
				for (j, w) in enumerate(data.sim_data.w_conductivity_range)
					Lxx_w, Lxy_w, Lyx_w, Lyy_w = current_correlation_K_4dir(data, w; pre_Gx=data.temporary_data.G_matrix_vec_x, pre_Gy=data.temporary_data.G_matrix_vec_y, δ=δ_fixed)
					measurements.Lxx_w_fixed[i][d_idx][j] = Lxx_w
					measurements.Lxy_w_fixed[i][d_idx][j] = Lxy_w
					measurements.Lyx_w_fixed[i][d_idx][j] = Lyx_w
					measurements.Lyy_w_fixed[i][d_idx][j] = Lyy_w
				end
			end
		end

		# DC conductivity (delta->0 integrated formula) for multiple Δω windows
		measure_conductivity_dc!(data, i; pre_Gx=data.temporary_data.G_matrix_vec_x, pre_Gy=data.temporary_data.G_matrix_vec_y)

		# L_D DC (delta->0 integrated current-current correlator) for multiple Δω windows
		measure_L_D_dc!(data, i; pre_Gx=data.temporary_data.G_matrix_vec_x, pre_Gy=data.temporary_data.G_matrix_vec_y)

		# L_D direct (ω→0, δ→0 closed-form current-current correlator)
		measure_L_D_direct!(data, i; pre_Gx=data.temporary_data.G_matrix_vec_x, pre_Gy=data.temporary_data.G_matrix_vec_y)

		# DdaggerD - <Delta^dagger * Delta>
		measurements.DdD[i] = use_fast ? compute_D_dagger_D_fast!(work, data) : compute_D_dagger_D_K(data)

		# Ps_l
		for (l_ix, l) in enumerate(data.const_struct.ls_list)
			## Ps_l - <Δ_i * Δ_i+l^dagger> - pairing correlation function at real space
			measurements.Ps_l[i][l_ix] = compute_Ps_l_K(data, l)
		end

		# Susceptibility measurement (spin and charge) in one shared pass
		susceptibility_q_vec_both!(data, data.const_struct.q_list_sus, measurements.chi_spin[i], measurements.chi_charge[i])

		# phase correlation
		has_dwave_observables = !isempty(measurements.Phase_Corr_dwave_x)
		for (l_ix, l) in enumerate(data.const_struct.ls_list_reduced)
			measurements.Phase_Corr[i][l_ix] = compute_phase_correlation(data, l; pairing_field=:Delta_s)
			if has_dwave_observables
				measurements.Phase_Corr_dwave_x[i][l_ix] = compute_phase_correlation(data, l; pairing_field=:Delta_dx)
				measurements.Phase_Corr_dwave_y[i][l_ix] = compute_phase_correlation(data, l; pairing_field=:Delta_dy)
				measurements.Phase_Corr_mixed[i][l_ix] = compute_mixed_phase_correlation(data, l; pairing_field1=:Delta_dx, pairing_field2=:Delta_dy)
			end
		end

		# Delta mean - mean of the Delta field
		measurements.Delta_mean[i] = mean(data.current_state.Delta_s)

		#Delta operator - Expectation value of <c_i,down c_i,up> over all sites
		measurements.Delta_operator[i] = compute_Delta_operator(data)

		# Delta_MC and Delta_MC_abs
		measurements.Delta_MC_abs[i][1] = mean(abs.(data.current_state.Delta_s))
		if length(measurements.Delta_MC_abs[i]) >= 3
			measurements.Delta_MC_abs[i][2] = mean(abs.(data.current_state.Delta_dx))
			measurements.Delta_MC_abs[i][3] = mean(abs.(data.current_state.Delta_dy))
		end

		measurements.Delta_MC[i][1] = mean(data.current_state.Delta_s)
		if length(measurements.Delta_MC[i]) >= 3
			measurements.Delta_MC[i][2] = mean(data.current_state.Delta_dx)
			measurements.Delta_MC[i][3] = mean(data.current_state.Delta_dy)
		end

		# MCMF action of K0
		measurements.U4[i] = data.current_state.S_K0

		# density-channel (phi) observables. All zero when the channel is off.
		phi = data.current_state.phi
		dsign = data.const_struct.density_sign
		N = data.sim_data.N
		measurements.phi_mean[i] = mean(phi)
		measurements.phi_abs_mean[i] = mean(abs, phi)
		measurements.phi_Q[i] = sum(k -> dsign[k] * phi[k], 1:N) / N

		# Electron charge-density-wave order parameter. Only meaningful for Q != 0
		# (for Q=0 it is just the density n), so skip the extra work otherwise.
		if any(!=(0.0), data.sim_data.Q_density)
			measurements.m_cdw[i] = staggered_density_K(data)
		end

		# --- MC-field histograms (lifetime accumulators; not part of MeasureData) ---
		# One click per site: phi_i into the 1D phi histogram, and (|Delta_s_i|,
		# arg(Delta_s_i)) into the 2D JOINT Delta histogram (amplitude x phase).
		# Out-of-range values clamp into the edge bins. nbins == 0 (empty accumulator)
		# turns a histogram off. Delta uses the s-wave field for now.
		cs = data.current_state
		sd = data.sim_data
		if sd.phi_hist_nbins > 0
			lo, hi = sd.phi_hist_range[1], sd.phi_hist_range[2]
			nb = sd.phi_hist_nbins
			@inbounds for k in 1:N
				cs.phi_hist[hist_bin_clamped(phi[k], lo, hi, nb)] += 1
			end
		end
		if sd.Delta_hist_abs_nbins > 0 && sd.Delta_hist_arg_nbins > 0
			ds = cs.Delta_s
			abs_lo, abs_hi = sd.Delta_hist_abs_range[1], sd.Delta_hist_abs_range[2]
			abs_nb = sd.Delta_hist_abs_nbins
			arg_nb = sd.Delta_hist_arg_nbins
			twopi = 2.0 * pi
			@inbounds for k in 1:N
				z = ds[k]
				ai = hist_bin_clamped(abs(z), abs_lo, abs_hi, abs_nb)          # amplitude bin (row)
				pj = hist_bin_clamped(mod(angle(z), twopi), 0.0, twopi, arg_nb) # phase bin (col)
				cs.Delta_hist[ai, pj] += 1
			end
		end

		return measurements
	end

	"""
	    make_measurement_n_only!(data, i)

	Cheap measurement for the μ search (occupation_convergence.jl): only n and the
	action are stored. Nothing else is measured and the large measurement buffers are
	never allocated.
	"""
	function make_measurement_n_only!(data::LatticeData, i::Int64)
		if i >data.sim_data.saving_interval
			i = data.sim_data.saving_interval
		end
		ensure_global_diag_synced!(data; reason="measurement_n_only")
		get_FD_dist!(data)
		measurements = data.measurement_data
		measurements.Ss[i] = data.current_state.S
		measurements.n[i] = avrage_n_K(data)
		return measurements
	end


	# ════════════════════════════════════════════════════════════════════════
	#  Fast measurement kernels (make_measurement_K!(...; fast=true), SC_L = 1)
	#
	#  Each kernel gives the same numbers as the reference function named in its
	#  comment, up to the order of floating-point summation
	#  (tests/test_measurement_kernels.jl). The reference functions are kept and
	#  run with fast=false. Why each one is faster: Obsidian note "Runtime profile
	#  and measurement speed-ups 06-10-2026".
	#
	#  Buffers: temporary_data.meas_work (MeasurementWork), built once per run by
	#  measurement_work!, plus G_matrix_vec_x/_y (ensure_G_buffers!). After that a
	#  measurement allocates nothing in these kernels except the task bookkeeping
	#  of the threaded loops.
	#
	#  Threads: per-task buffers are indexed by CHUNK number, never by
	#  Threads.threadid(): Julia 1.12 adds an interactive thread, so thread ids
	#  are not 1:nthreads().
	# ════════════════════════════════════════════════════════════════════════

	"""
	    measurement_G_length(sim_data) -> Int

	Number of 2N×2N current-vertex matrices kept per direction: max(Lx, Ly) + 1 (one per q).
	"""
	measurement_G_length(sim_data::SimData) = max(max(sim_data.Lx, sim_data.Ly) + 1, sim_data.SC_L^2)

	"""
	    ensure_G_buffers!(data)

	Allocate the current-vertex matrices `G_matrix_vec_x/_y` (`measurement_G_length`
	matrices of 2N×2N each; 4.2 GB at L = 32) on first use. They are grown in place
	(`push!`), so a caller that already holds the vector sees the new matrices.
	"""
	function ensure_G_buffers!(data::LatticeData)
		td =data.temporary_data
		N = data.sim_data.N
		G_length = measurement_G_length(data.sim_data)
		for G_vec in (td.G_matrix_vec_x, td.G_matrix_vec_y)
			while length(G_vec) < G_length
				push!(G_vec, zeros(ComplexF64, 2*N, 2*N))
			end
		end
		return nothing
	end

	"""
	    build_current_vertex_op(data, direction) -> CurrentVertexOp

	q-independent data of the +a current operator, a = `direction` ("x"/"y"): the four
	terms per site of `build_G_r_from_S_r_x!`/`y!` (n = i + a, T = boundary twist ×
	Peierls phase of the bond):

	    M[n,   i  ] =  conj(T)      M[i,   n  ] = -T
	    M[N+i, N+n] = -conj(T)      M[N+n, N+i] =  T

	so that G^a(q) = U' M(q) U, with M(q) = these terms times e^{i q·r_i}.
	"""
	function build_current_vertex_op(data::LatticeData, direction::String)
		cs =data.const_struct
		N = data.sim_data.N
		if direction == "x"
			nb, bc, L_dir, col = cs.new_ix_list, cs.x, data.sim_data.Lx, 1
		elseif direction == "y"
			nb, bc, L_dir, col = cs.new_iy_list, cs.y, data.sim_data.Ly, 2
		else
			error("direction must be 'x' or 'y'")
		end
		# same phases as build_G_r_from_S_r_x!/y! (K = 1)
		phase_Tr = exp(-1im * 2π/L_dir * cs.SC_K_vectors[1][col])
		Tr = [(bc[i] == L_dir ? phase_Tr : (1.0 + 0.0im)) * cs.peierls_phases[i, col] for i in 1:N]

		# sparse pattern for the per-q GEMM; duplicate (row, col) pairs (L = 2) are merged
		I = zeros(Int64, 4N)
		J = zeros(Int64, 4N)
		for i in 1:N
			n = nb[i]
			I[4i-3] = n;     J[4i-3] = i
			I[4i-2] = i;     J[4i-2] = n
			I[4i-1] = N + i; J[4i-1] = N + n
			I[4i]   = N + n; J[4i]   = N + i
		end
		M = sparse(I, J, ones(ComplexF64, 4N), 2N, 2N)
		slot = zeros(Int64, 4N)
		for k in 1:4N
			r = M.colptr[J[k]]:(M.colptr[J[k]+1] - 1)
			slot[k] = r[findfirst(==(I[k]), view(M.rowval, r))]
		end

		# row / column blocks for the row-block method
		groups = CurrentVertexGroup[]
		for (axis, key) in ((:y, cs.y), (:x, cs.x))
			for g in sort(unique(key))
				S = findall(==(g), key)
				R = unique(vcat(S, nb[S]))
				rows = vcat(R, R .+ N)
				pos = Dict(r => k for (k, r) in enumerate(rows))
				Mg = zeros(ComplexF64, length(rows), length(rows))
				for i in S
					n = nb[i]
					Mg[pos[n],     pos[i]]     +=  conj(Tr[i])
					Mg[pos[i],     pos[n]]     += -Tr[i]
					Mg[pos[N + i], pos[N + n]] += -conj(Tr[i])
					Mg[pos[N + n], pos[N + i]] +=  Tr[i]
				end
				push!(groups, CurrentVertexGroup(axis, g, rows, Mg))
			end
		end
		return CurrentVertexOp(M, slot, Tr, groups)
	end

	"""
	    MeasurementWork(data) -> MeasurementWork

	Allocate the fast-kernel buffers for this lattice and these ω/δ grids, and build the
	current-vertex operators and the Fourier rows of the spectral-function k-path and BZ
	map. Requires `SC_L = 1`.
	"""
	function MeasurementWork(data::LatticeData)
		sd = data.sim_data
		cs = data.const_struct
		sd.SC_L == 1 || error("MeasurementWork: the fast measurement kernels need SC_L = 1")
		N = sd.N
		twoN = 2 * N
		n_w = length(sd.w_list)
		n_w_cond = length(sd.w_conductivity_range)
		n_delta = length(sd.delta_values_fixed)
		n_chunks = Threads.nthreads()
		n_q = length(cs.q_list_CC)

		vertex_x = build_current_vertex_op(data, "x")
		vertex_y = build_current_vertex_op(data, "y")
		r_max = maximum(length(g.rows) for g in vcat(vertex_x.groups, vertex_y.groups))

		Mlm = zeros(Float64, twoN, twoN)
		pair_dE = vec(Mlm)               # same memory: L_ab(q) is done before fixed-delta packs

		# Fourier rows of the spectral-function k-path and of the BZ map, built
		# exactly as spectral_functionV2 builds its FT vector: e^{-i (k + K).r_n}
		kx_path, ky_path = cs.SF_k_path
		n_path = length(kx_path)
		F_path = zeros(ComplexF64, n_path, N)
		for p in 1:n_path
			k_gen = cs.k_general_idx[kx_path[p], ky_path[p]]
			q_ix, K_ix = cs.k_general_qK_idx[k_gen]
			kx, ky = cs.k_vectors[q_ix]
			Kx, Ky = cs.SC_K_vectors[K_ix]
			for n in 1:N
				F_path[p, n] = exp(-1im * ((kx + Kx) * cs.x[n] + (ky + Ky) * cs.y[n]))
			end
		end
		N_gen = length(cs.k_general_qK_idx)
		F_all = zeros(ComplexF64, N_gen, N)
		for gen_ix in 1:N_gen
			q_ix, K_ix = cs.k_general_qK_idx[gen_ix]
			kx, ky = cs.k_vectors[q_ix]
			Kx, Ky = cs.SC_K_vectors[K_ix]
			for n in 1:N
				F_all[gen_ix, n] = exp(-1im * ((kx + Kx) * cs.x[n] + (ky + Ky) * cs.y[n]))
			end
		end

		return MeasurementWork(N, n_w, n_w_cond, n_delta, n_chunks,
			vertex_x, vertex_y,
			zeros(ComplexF64, r_max, twoN), zeros(ComplexF64, r_max, twoN),   # UR, MU
			zeros(ComplexF64, twoN, twoN),                                     # scratch
			Mlm, zeros(ComplexF64, 4, n_q),                                    # Mlm, Lab_q
			pair_dE, zeros(Float64, twoN^2), zeros(Float64, twoN^2),           # pair_dE, pair_xx, pair_yy
			zeros(Float64, twoN^2), zeros(Float64, twoN^2),                    # pair_xy_re, pair_xy_im
			zeros(Float64, 8, n_delta * n_w_cond, max(n_chunks, 1)),           # fd_acc
			F_path, F_all, zeros(Float64, twoN, n_w),                          # F_path, F_all, Lor
			zeros(ComplexF64, n_path, twoN), zeros(Float64, n_path, twoN),     # FU_path, W_path
			zeros(Float64, N_gen, twoN),                                       # W_N2N
			zeros(Float64, n_path, n_w), zeros(Float64, N_gen), zeros(Float64, N, n_w))  # A_path, BZ, LDOS
	end

	"""
	    measurement_work_fits(work, data) -> Bool

	True if `work` was sized for this lattice, these ω/δ grids, q list and thread count.
	"""
	function measurement_work_fits(work::MeasurementWork, data::LatticeData)
		sd = data.sim_data
		return work.N == sd.N && work.n_w == length(sd.w_list) &&
			work.n_w_cond == length(sd.w_conductivity_range) &&
			work.n_delta == length(sd.delta_values_fixed) &&
			work.n_chunks == Threads.nthreads() &&
			size(work.Lab_q, 2) == length(data.const_struct.q_list_CC)
	end

	"""
	    measurement_work!(data) -> MeasurementWork

	The fast-kernel buffers, built on first use and rebuilt if the lattice or the ω/δ
	grids no longer match what they were sized for.
	"""
	function measurement_work!(data::LatticeData)
		td =data.temporary_data
		work = td.meas_work
		if work === nothing || !measurement_work_fits(work, data)
			td.meas_work = nothing   # let the GC take the old buffers first
			work = MeasurementWork(data)
			td.meas_work = work
		end
		return work::MeasurementWork
	end

	"""
	    ensure_measurement_buffers!(data; fast=true)

	Allocate everything a full measurement needs. `MonteCarloSimulation` calls this
	before thermalization, so a job that cannot hold them fails at once.
	"""
	function ensure_measurement_buffers!(data::LatticeData; fast::Bool=true)
		ensure_G_buffers!(data)
		if fast && data.sim_data.SC_L == 1
			measurement_work!(data)
		end
		return nothing
	end


	# ── current vertices G^a(q)[l,m] = <l| j^a(q) |m>   (replaces make_G_matrix_q_vec) ──

	"""
	    current_vertices_rowblock!(G_out, work, data, q_vecs, op)

	Current vertices G^a(q) by the row-block method; every q must lie on an axis
	(`q_list_CC` does). For the (0, qy) family the phase e^{i qy y_i} is constant along a
	lattice row, so

	    G(0, qy) = Σ_y e^{i qy y} G_row(y),   G_row(y) = U[R_y,:]' M_y U[R_y,:]

	with R_y the up/down rows of the sites of row y and their neighbours. The (qx, 0)
	family is the same with columns. All the row GEMMs together cost about 24 N³ per
	direction, then each q is an axpy, instead of 8 N³ per q.
	"""
	function current_vertices_rowblock!(G_out, work::MeasurementWork, data::LatticeData, q_vecs, op::CurrentVertexOp)
		N =data.sim_data.N
		twoN = 2 * N
		U = view(data.current_state.U_matrix, :, :, 1)
		q_length = length(q_vecs)
		for iq in 1:q_length
			q = q_vecs[iq]
			(q[1] == 0.0 || q[2] == 0.0) || error("current_vertices_rowblock!: q = $(q) is not on an axis")
			fill!(G_out[iq], 0)
		end
		G_row = work.scratch
		for grp in op.groups
			r = length(grp.rows)
			UR = view(work.UR, 1:r, :)
			MU = view(work.MU, 1:r, :)
			rows = grp.rows
			@inbounds for c in 1:twoN
				for k in 1:r
					UR[k, c] = U[rows[k], c]
				end
			end
			mul!(MU, grp.Mg, UR)
			mul!(G_row, UR', MU)
			for iq in 1:q_length
				q = q_vecs[iq]
				if grp.axis === :y
					q[1] == 0.0 || continue                   # (0, qy) family, q = 0 included
					axpy!(cis(q[2] * grp.coord), G_row, G_out[iq])
				else
					q[1] != 0.0 || continue                   # (qx, 0) family
					axpy!(cis(q[1] * grp.coord), G_row, G_out[iq])
				end
			end
		end
		return G_out
	end

	"""
	    current_vertices_gemm!(G_out, work, data, q_vecs, op)

	Current vertices G^a(q) = U' M(q) U for any q: one sparse × dense product and one
	ZGEMM per q. The sparsity pattern of M is fixed; per q only its 4N values are rewritten.
	"""
	function current_vertices_gemm!(G_out, work::MeasurementWork, data::LatticeData, q_vecs, op::CurrentVertexOp)
		N =data.sim_data.N
		U = view(data.current_state.U_matrix, :, :, 1)
		FT = get_FT_table_for_q_vecs!(data, q_vecs)          # FT[i, iq] = e^{+i q.r_i}
		M = op.M
		nz = nonzeros(M)
		slot = op.slot
		Tr = op.Tr
		T = work.scratch
		for iq in eachindex(q_vecs)
			fill!(nz, 0)
			@inbounds for i in 1:N
				ph = FT[i, iq]
				t = Tr[i]
				nz[slot[4i-3]] += ph * conj(t)
				nz[slot[4i-2]] -= ph * t
				nz[slot[4i-1]] -= ph * conj(t)
				nz[slot[4i]]   += ph * t
			end
			mul!(T, M, U)
			mul!(G_out[iq], U', T)
		end
		return G_out
	end

	"""
	    current_vertices_fast!(G_out, work, data, q_vecs, direction; method=:auto)

	Fill `G_out[iq]` with the current vertices G^a(q)[l,m] = ⟨l| j^a(q) |m⟩ in the BdG
	eigenbasis, the same matrices as `make_G_matrix_q_vec(data, q_vecs; direction)`.
	`method=:auto` uses the row-block method when every q is on an axis (the full
	`q_list_CC`) and the per-q GEMM otherwise.
	"""
	function current_vertices_fast!(G_out, work::MeasurementWork, data::LatticeData, q_vecs, direction::String; method::Symbol=:auto)
		op =direction == "x" ? work.vertex_x : direction == "y" ? work.vertex_y : error("direction must be 'x' or 'y'")
		length(G_out) >= length(q_vecs) || error("G_out holds $(length(G_out)) matrices, $(length(q_vecs)) needed")
		if method === :auto
			method = all(q -> q[1] == 0.0 || q[2] == 0.0, q_vecs) ? :rowblock : :gemm
		end
		if method === :rowblock
			return current_vertices_rowblock!(G_out, work, data, q_vecs, op)
		elseif method === :gemm
			return current_vertices_gemm!(G_out, work, data, q_vecs, op)
		end
		error("unknown current-vertex method $(method) (use :auto, :rowblock or :gemm)")
	end


	# ── L_ab(q, w=0)   (replaces the 4 current_correlation_from_G! calls) ──

	"""
	    current_correlation_q_fast!(out, work, data, Gx, Gy, q_length) -> out

	Static current-current correlations Λab(q, ω = 0) for all q: `out[k, iq]`, rows
	k = (xx, xy, yx, yy), equals `Lxx_temp[iq]` of the k-th `current_correlation_from_G!` call:

	    Λab(q) = (t²/N) [ -Σ_lm G^a_lm conj(G^b_lm) M_lm + β (Σ_l G^a_ll f_l) conj(Σ_l G^b_ll f_l) ]

	with M_lm = (f_l - f_m)/(E_l - E_m) (β f_l (f_l - 1) for equal energies). M_lm is built
	once per measurement, the 4 components are summed in one pass, and q is threaded.
	"""
	function current_correlation_q_fast!(out::AbstractMatrix{ComplexF64}, work::MeasurementWork, data::LatticeData, Gx, Gy, q_length::Int)
		f =view(data.temporary_data.f, :, 1)
		E = view(data.current_state.Energies, :, 1)
		beta = data.sim_data.beta
		N = data.sim_data.N
		twoN = 2 * N
		Mlm = work.Mlm
		@inbounds for m in 1:twoN
			for l in 1:twoN
				diff = E[l] - E[m]
				Mlm[l, m] = abs(diff) < 1e-12 ? beta * f[l] * (f[l] - 1) : (f[l] - f[m]) / diff
			end
		end
		prefactor = data.sim_data.t^2 / N
		Threads.@threads for iq in 1:q_length
			gx = Gx[iq]
			gy = Gy[iq]
			sxx = 0.0
			syy = 0.0
			sxy = 0.0im
			dx = 0.0im
			dy = 0.0im
			@inbounds for m in 1:twoN
				for l in 1:twoN
					a = gx[l, m]
					b = gy[l, m]
					w = Mlm[l, m]
					sxx += abs2(a) * w
					syy += abs2(b) * w
					sxy += a * conj(b) * w
				end
				dx += gx[m, m] * f[m]
				dy += gy[m, m] * f[m]
			end
			@inbounds begin
				out[1, iq] = prefactor * (-sxx + beta * abs2(dx))
				out[2, iq] = prefactor * (-sxy + beta * dx * conj(dy))
				out[3, iq] = prefactor * (-conj(sxy) + beta * dy * conj(dx))
				out[4, iq] = prefactor * (-syy + beta * abs2(dy))
			end
		end
		return out
	end

	"""
	    measure_current_correlations_q_fast!(data, measurements, i, work; vertex_method=:auto) -> (Gx, Gy)

	Fast counterpart of `measure_current_correlations_q!(...; return_Gs=true)`: build
	G^x(q), G^y(q) for `q_list_CC` into `G_matrix_vec_x/_y` and write the Lab_q0, Lab_qy
	and Lab_qx entries of measurement `i`.
	"""
	function measure_current_correlations_q_fast!(data::LatticeData, measurements::MeasureData, i::Int, work::MeasurementWork; vertex_method::Symbol=:auto)
		td =data.temporary_data
		q_list_CC = data.const_struct.q_list_CC
		q_length = length(q_list_CC)
		n_qy = div(data.sim_data.Ly * data.sim_data.SC_L, 2)
		n_qx = div(data.sim_data.Lx * data.sim_data.SC_L, 2)
		G_x_vec = td.G_matrix_vec_x
		G_y_vec = td.G_matrix_vec_y
		current_vertices_fast!(G_x_vec, work, data, q_list_CC, "x"; method=vertex_method)
		current_vertices_fast!(G_y_vec, work, data, q_list_CC, "y"; method=vertex_method)
		L = current_correlation_q_fast!(work.Lab_q, work, data, G_x_vec, G_y_vec, q_length)

		# index 1 -> q = 0; next n_qy -> (0, qy); last n_qx -> (qx, 0)
		measurements.Lxx_q0[i] = real(L[1, 1])
		measurements.Lxy_q0[i] = L[2, 1]
		measurements.Lyx_q0[i] = L[3, 1]
		measurements.Lyy_q0[i] = real(L[4, 1])
		for k in 1:n_qy
			measurements.Lxx_qy[i][k] = real(L[1, 1 + k])
			measurements.Lxy_qy[i][k] = L[2, 1 + k]
			measurements.Lyx_qy[i][k] = L[3, 1 + k]
			measurements.Lyy_qy[i][k] = real(L[4, 1 + k])
		end
		for k in 1:n_qx
			measurements.Lxx_qx[i][k] = real(L[1, 1 + n_qy + k])
			measurements.Lxy_qx[i][k] = L[2, 1 + n_qy + k]
			measurements.Lyx_qx[i][k] = L[3, 1 + n_qy + k]
			measurements.Lyy_qx[i][k] = real(L[4, 1 + n_qy + k])
		end
		return G_x_vec, G_y_vec
	end


	# ── fixed-delta conductivity L_ab(w + i delta)   (replaces the current_correlation_K_4dir grid) ──

	"""
	    fixed_delta_block_sums(work, p0, p1, w, δ) -> NTuple{8, Float64}

	Sum over the packed eigen-pairs p0:p1 of 1/(w + ΔE + iδ) times the pair weights.
	Returns (re, im) of xx, yy, xy, yx.
	"""
	@inline function fixed_delta_block_sums(work::MeasurementWork, p0::Int, p1::Int, w::Float64, δ::Float64)
		dE =work.pair_dE
		Pxx = work.pair_xx
		Pyy = work.pair_yy
		Pr = work.pair_xy_re
		Pi = work.pair_xy_im
		δ2 = δ * δ
		axx_r = 0.0; axx_i = 0.0; ayy_r = 0.0; ayy_i = 0.0
		axy_r = 0.0; axy_i = 0.0; ayx_r = 0.0; ayx_i = 0.0
		@inbounds @simd for p in p0:p1
			a = w + dE[p]
			inv = 1.0 / (a * a + δ2)
			re = a * inv
			im = δ * inv
			axx_r += Pxx[p] * re;  axx_i -= Pxx[p] * im
			ayy_r += Pyy[p] * re;  ayy_i -= Pyy[p] * im
			axy_r += Pr[p] * re + Pi[p] * im;   axy_i += Pi[p] * re - Pr[p] * im
			ayx_r += Pr[p] * re - Pi[p] * im;   ayx_i += -Pi[p] * re - Pr[p] * im
		end
		return (axx_r, axx_i, ayy_r, ayy_i, axy_r, axy_i, ayx_r, ayx_i)
	end

	"""
	    measure_fixed_delta_fast!(data, measurements, i, work, Gx0, Gy0; block=4096) -> Float64

	Λab(ω + iδ) at q = 0 for every broadening δ in `delta_values_fixed` and every ω in
	`w_conductivity_range`: writes `Lab_w_fixed[i][d][j]`, the same as
	`current_correlation_K_4dir(data, ω_j; pre_Gx, pre_Gy, δ=δ_d)`. `Gx0`, `Gy0` are the
	q = 0 vertices. Returns the fraction of eigen-pairs kept.

	1. Pack: the kernel is (f_l - f_m)/(ω + ΔE + iδ), so each pair needs only ΔE and the
	   weights Δf|gx|², Δf|gy|², Δf gx conj(gy), written once into contiguous arrays.
	   Pairs with Δf == 0 exactly are dropped.
	2. Block: the pairs are processed in blocks of `block` (160 KB for 4096) and each
	   block is used for every (δ, ω) while it is in cache.
	3. Threads: blocks are dealt round-robin to the chunks, each with its own
	   accumulators; chunk sums are reduced in a fixed order.
	"""
	function measure_fixed_delta_fast!(data::LatticeData, measurements::MeasureData, i::Int, work::MeasurementWork,
									   Gx0::AbstractMatrix{ComplexF64}, Gy0::AbstractMatrix{ComplexF64}; block::Int=4096)
		f =view(data.temporary_data.f, :, 1)
		E = view(data.current_state.Energies, :, 1)
		N = data.sim_data.N
		twoN = 2 * N
		ws = data.sim_data.w_conductivity_range
		ds = data.sim_data.delta_values_fixed
		nd = length(ds)
		n_grid = nd * length(ws)
		n_grid == 0 && return 0.0

		# 1) pack, column-major
		np = 0
		@inbounds for m in 1:twoN
			fm = f[m]
			Em = E[m]
			for l in 1:twoN
				df = f[l] - fm
				df == 0.0 && continue
				np += 1
				gx = Gx0[l, m]
				gy = Gy0[l, m]
				gxy = gx * conj(gy)
				work.pair_dE[np] = E[l] - Em
				work.pair_xx[np] = df * abs2(gx)
				work.pair_yy[np] = df * abs2(gy)
				work.pair_xy_re[np] = df * real(gxy)
				work.pair_xy_im[np] = df * imag(gxy)
			end
		end

		# 2+3) cache-blocked accumulation. `n_pairs` is bound once: capturing the
		# reassigned `np` in the threaded closure would box it (Core.Box) and make
		# every kernel call a dynamic dispatch.
		n_pairs = np
		n_blocks = cld(n_pairs, block)
		acc = work.fd_acc
		n_chunks = size(acc, 3)
		fill!(acc, 0.0)
		Threads.@threads for c in 1:n_chunks
			for ib in c:n_chunks:n_blocks
				p0 = (ib - 1) * block + 1
				p1 = min(ib * block, n_pairs)
				@inbounds for k in 1:n_grid
					s = fixed_delta_block_sums(work, p0, p1, ws[cld(k, nd)], ds[mod1(k, nd)])
					for a in 1:8
						acc[a, k, c] += s[a]
					end
				end
			end
		end

		# reduce over chunks and write (xx, xy, yx, yy)
		pref = -data.sim_data.t^2 / N
		@inbounds for k in 1:n_grid
			sxx_r = 0.0; sxx_i = 0.0; syy_r = 0.0; syy_i = 0.0
			sxy_r = 0.0; sxy_i = 0.0; syx_r = 0.0; syx_i = 0.0
			for c in 1:n_chunks
				sxx_r += acc[1, k, c]; sxx_i += acc[2, k, c]
				syy_r += acc[3, k, c]; syy_i += acc[4, k, c]
				sxy_r += acc[5, k, c]; sxy_i += acc[6, k, c]
				syx_r += acc[7, k, c]; syx_i += acc[8, k, c]
			end
			d_idx = mod1(k, nd)
			j = cld(k, nd)
			measurements.Lxx_w_fixed[i][d_idx][j] = pref * complex(sxx_r, sxx_i)
			measurements.Lxy_w_fixed[i][d_idx][j] = pref * complex(sxy_r, sxy_i)
			measurements.Lyx_w_fixed[i][d_idx][j] = pref * complex(syx_r, syx_i)
			measurements.Lyy_w_fixed[i][d_idx][j] = pref * complex(syy_r, syy_i)
		end
		return n_pairs / twoN^2
	end


	# ── <Delta^dagger Delta>   (replaces compute_D_dagger_D_K) ──

	"""
	    compute_D_dagger_D_fast!(work, data) -> Float64

	⟨Δ†Δ⟩/N of the pair operator Δ = Σ_i c_i↓ c_i↑ (fast version of
	`compute_D_dagger_D_K`). With V = U_up' U_dn (one ZGEMM):

	    DdD = ( Σ_lm f_l (1 - f_m) |V_lm|² + |Σ_l f_l V_ll|² ) / N
	"""
	function compute_D_dagger_D_fast!(work::MeasurementWork, data::LatticeData)
		N =data.sim_data.N
		twoN = 2 * N
		U = view(data.current_state.U_matrix, :, :, 1)
		f = view(data.temporary_data.f, :, 1)
		V = work.scratch
		mul!(V, view(U, 1:N, :)', view(U, (N+1):twoN, :))
		DdD = 0.0
		add2_term = 0.0im
		@inbounds for m in 1:twoN
			one_minus_fm = 1 - f[m]
			for l in 1:twoN
				DdD += f[l] * one_minus_fm * abs2(V[l, m])
			end
			add2_term += f[m] * V[m, m]
		end
		return (DdD + abs2(add2_term)) / N
	end


	# ── spectral function on the k-path, BZ map at w = 0, LDOS   (replaces the w_list loop) ──

	"""
	    spectral_fast!(work, data) -> work

	Fill `work.A_path[p, j]` (= `spectral_functionV2` at the p-th k-path point, ω_j),
	`work.BZ` (= `spectral_function_fullV2(data, 0.0)`, only when 0.0 is in `w_list`) and
	`work.LDOS[:, j]` (= `compute_full_LDOS(data, ω_j)`), all with Lorentzian width
	`delta_max`. The projections |⟨k|u_l⟩|² do not depend on ω: they are built once and
	every ω is one GEMM against the table of Lorentzians `Lor[l, j]`.
	"""
	function spectral_fast!(work::MeasurementWork, data::LatticeData)
		N =data.sim_data.N
		twoN = 2 * N
		U = view(data.current_state.U_matrix, :, :, 1)
		E = view(data.current_state.Energies, :, 1)
		U_up = view(U, 1:N, :)
		w_list = data.sim_data.w_list
		delt = data.temporary_data.delta_max
		Lor = work.Lor
		@inbounds for j in eachindex(w_list)
			for l in 1:twoN
				Lor[l, j] = (1 / (pi * delt)) / (1 + ((E[l] - w_list[j]) / delt)^2)
			end
		end
		# k-path
		mul!(work.FU_path, work.F_path, U_up)
		work.W_path .= abs2.(work.FU_path)
		mul!(work.A_path, work.W_path, Lor)
		rmul!(work.A_path, 1 / N)
		# BZ map at w = 0 (F_all * U_up goes into the shared 2N x 2N scratch)
		iw0 = findfirst(==(0.0), w_list)
		if iw0 !== nothing
			FU_all = view(work.scratch, 1:size(work.F_all, 1), :)
			mul!(FU_all, work.F_all, U_up)
			work.W_N2N .= abs2.(FU_all)
			mul!(work.BZ, work.W_N2N, view(Lor, :, iw0))
			rmul!(work.BZ, 1 / N)
		end
		# LDOS (factor 2: spin); W_N2N now holds |U_up|^2
		work.W_N2N .= abs2.(U_up)
		mul!(work.LDOS, work.W_N2N, Lor)
		rmul!(work.LDOS, 2)
		return work
	end

	"""
	    measure_spectral_fast!(data, measurements, i, work)

	Update the running means of A(k, ω), BZ and LDOS and store DOS(ω) of measurement `i`,
	from `spectral_fast!` (same as the reference ω loop of `make_measurement_K!`).
	"""
	function measure_spectral_fast!(data::LatticeData, measurements::MeasureData, i::Int, work::MeasurementWork)
		spectral_fast!(work, data)
		w_list = data.sim_data.w_list
		N_gen = data.sim_data.N * data.sim_data.SC_L^2
		n_path = size(work.A_path, 1)
		for j in eachindex(w_list)
			#LDOS
			LDOS_c = view(work.LDOS, :, j)
			measurements.LDOS[j] .*= i-1
			measurements.LDOS[j] .+= LDOS_c
			measurements.LDOS[j] ./= i

			# DOS
			measurements.DOS[i][j] += sum(LDOS_c) / N_gen

			# Spectral function
			A_j = measurements.A[j]
			for l in 1:n_path
				A_j[l] *= i-1
				A_j[l] += work.A_path[l, j]
				A_j[l] /= i
			end

			if w_list[j] == 0.0
				# BZ - Brillouin zone (once per 0.0 entry, as the reference loop)
				measurements.BZ .*= i-1
				measurements.BZ .+= work.BZ
				measurements.BZ ./= i
			end
		end
		return nothing
	end


	# ════════════════════════════════════════════════════════════════════════
	#  Memory check: estimated need of a run vs the memory the job may use
	# ════════════════════════════════════════════════════════════════════════

	# Julia runtime + loaded packages (MKL, Plots, JLD2, ...) + compiled code: peak
	# RSS minus the live arrays of a short run, 0.90-1.03 GB at L=12-24 (laptop,
	# Julia 1.10, 2026-10-06). bench_profile_run.jl "integ" reports it on the clusters.
	const MEMORY_RUNTIME_OVERHEAD_GB = 1.1
	# allowance for garbage-collector slack and short-lived copies
	const MEMORY_HEADROOM = 0.10

	"""
	    measurement_work_bytes(sim_data) -> Int

	Bytes of `MeasurementWork` for these parameters (mirrors `MeasurementWork(data)`).
	"""
	function measurement_work_bytes(sim_data::SimData)
		Lx =sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		twoN = 2 * N
		n_q = 1 + div(Lx, 2) + div(Ly, 2)
		n_w = length(sim_data.w_list)
		n_wc = length(sim_data.w_conductivity_range)
		n_d = length(sim_data.delta_values_fixed)
		n_path = 2 * (Lx - div(Lx, 2)) + (Ly - div(Ly, 2)) + 1
		c = sizeof(ComplexF64)
		r = sizeof(Float64)
		b = twoN^2 * c                                   # scratch
		b += 5 * twoN^2 * r                              # Mlm (= pair_dE) + 4 packed pair arrays
		b += 8 * n_d * n_wc * Threads.nthreads() * r + 4 * n_q * c
		# vertex operators of both directions: sparse pattern, slot, Tr ...
		b += 2 * (4N * (c + 8) + (twoN + 1) * 8 + 4N * 8 + N * c)
		# ... and row/column blocks: x op rows r=2Lx, columns r=4Ly; y op rows r=4Lx, columns r=2Ly
		b += (Ly * (2Lx)^2 + Lx * (4Ly)^2 + Ly * (4Lx)^2 + Lx * (2Ly)^2) * c
		b += 2 * 4 * max(Lx, Ly) * twoN * c              # UR, MU
		# spectral: F_path, F_all, Lor, FU_path + W_path, W_N2N, A_path, BZ, LDOS
		b += n_path * N * c + N * N * c + twoN * n_w * r + n_path * twoN * (c + r) +
			 N * twoN * r + n_path * n_w * r + N * r + N * n_w * r
		return b
	end

	"""
	    field_hist_bytes(sim_data) -> Int

	Bytes of the field-histogram accumulators (0.58 MB at φ 1000 bins + Δ 1500 × 50).
	Snapshots hold nothing between writes.
	"""
	function field_hist_bytes(sim_data::SimData)
		nphi =max(0, sim_data.phi_hist_nbins)
		na = max(0, sim_data.Delta_hist_abs_nbins)
		np = max(0, sim_data.Delta_hist_arg_nbins)
		return (nphi + (na > 0 && np > 0 ? na * np : 0)) * sizeof(Int64)
	end

	"""
	    estimate_memory_bytes(sim_data; measurements=true, fast=true)
	        -> (total, runtime, lattice, G_buffers, fast_work)

	Estimated peak memory of one run, in bytes. The large arrays are exact functions of
	the parameters; the Julia runtime and GC slack are the calibrated constants above.
	- `lattice`: U_matrix and its backup (per K), the K matrix, U_matrix_K, the
	  eigenvector workspace, the dense neighbour matrices, MeasureData, field histograms;
	- `G_buffers`: G_matrix_vec_x/_y and the susceptibility scratch (measurements only);
	- `fast_work`: fast-kernel buffers (measurements with fast=true, SC_L = 1).
	Estimate 1.18 / 1.69 / 2.46 GB vs measured peak RSS 1.10 / 1.43 / 2.15 GB at
	L = 12 / 20 / 24 (campaign grids, laptop, Julia 1.10).
	"""
	function estimate_memory_bytes(sim_data::SimData; measurements::Bool=true, fast::Bool=true)
		N =sim_data.N
		twoN = 2 * N
		N_SC = sim_data.SC_L^2
		mat2N = twoN^2 * sizeof(ComplexF64)
		matN = N^2 * sizeof(ComplexF64)
		lattice = (2 * N_SC + 3) * mat2N + (N_SC + 2) * matN + Base.summarysize(Initial_measurments(sim_data)) +
				  field_hist_bytes(sim_data)
		G = measurements ? 2 * measurement_G_length(sim_data) * mat2N + 2 * N * twoN * sizeof(ComplexF64) : 0
		work = (measurements && fast && sim_data.SC_L == 1) ? measurement_work_bytes(sim_data) : 0
		runtime = round(Int64, MEMORY_RUNTIME_OVERHEAD_GB * 2^30)
		total = runtime + round(Int64, (1 + MEMORY_HEADROOM) * (lattice + G + work))
		return (total=total, runtime=runtime, lattice=lattice, G_buffers=G, fast_work=work)
	end

	"""
	    job_memory_limit() -> (bytes, source)

	Memory this process may use: the SLURM allocation (the limit the job is killed at),
	else a cgroup limit, else physical memory. `SLURM_MEM_PER_CPU` wins over
	`SLURM_MEM_PER_NODE` (see the comment inside).
	"""
	function job_memory_limit()
		# SLURM sets only the variable matching the job's own request, so if both are
		# set, one was inherited: sbatch exports the submitting shell's environment,
		# and a job sbatch'ed from inside another job (pre_pre_run's --mem=4GB job)
		# arrives with that job's SLURM_MEM_PER_NODE=4096 next to its own
		# SLURM_MEM_PER_CPU. Every job template here requests --mem-per-cpu, so that
		# one wins. (2026-10-06: every Landau L=32 job refused itself at "4 GB" while
		# holding 8 GB.)
		mem_cpu = tryparse(Int64, strip(get(ENV, "SLURM_MEM_PER_CPU", "")))
		if mem_cpu !== nothing && mem_cpu > 0
			cpus = something(tryparse(Int64, strip(get(ENV, "SLURM_CPUS_PER_TASK", ""))), 1)
			return (mem_cpu * cpus * 2^20, "SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK")
		end
		mem_node = tryparse(Int64, strip(get(ENV, "SLURM_MEM_PER_NODE", "")))
		if mem_node !== nothing && mem_node > 0
			return (mem_node * 2^20, "SLURM_MEM_PER_NODE")
		end
		total = Int64(Sys.total_memory())
		if isdefined(Sys, :constrained_memory)
			c = Sys.constrained_memory()
			if 0 < c < total
				return (Int64(c), "cgroup limit")
			end
		end
		return (total, "physical memory")
	end

	"""
	    memory_check_skipped() -> Bool

	True if the environment variable `MC_SKIP_MEMORY_CHECK` is 1/true/yes.
	"""
	memory_check_skipped() = lowercase(strip(get(ENV, "MC_SKIP_MEMORY_CHECK", ""))) in ("1", "true", "yes")

	"""
	    check_memory_budget(sim_data; fast=true, verbose=false) -> estimate

	Stop with an error BEFORE the run starts if its estimated memory
	(`estimate_memory_bytes`) exceeds what the job may use (`job_memory_limit`).
	Override: environment variable `MC_SKIP_MEMORY_CHECK=1` (then only a warning).
	"""
	function check_memory_budget(sim_data::SimData; fast::Bool=true, verbose::Bool=false)
		est =estimate_memory_bytes(sim_data; fast=fast)
		limit, source = job_memory_limit()
		gb(x) = round(x / 2^30, digits=2)
		summary = "estimated $(gb(est.total)) GB [G_matrix_vec $(gb(est.G_buffers)), fast-kernel work $(gb(est.fast_work)), " *
				  "lattice $(gb(est.lattice)), runtime $(gb(est.runtime)), +$(round(Int, 100 * MEMORY_HEADROOM))% headroom] " *
				  "vs limit $(gb(limit)) GB ($(source))"
		if est.total > limit
			if memory_check_skipped()
				@warn "Memory check skipped (MC_SKIP_MEMORY_CHECK): $(summary)"
			else
				error("Not enough memory for Lx=$(sim_data.Lx), Ly=$(sim_data.Ly) (N=$(sim_data.N)): $(summary). " *
					  "Request more memory for the job, or set MC_SKIP_MEMORY_CHECK=1 to run anyway.")
			end
		elseif verbose
			println("Memory check: $(summary)")
		end
		return est
	end



	"""
	    make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; K=1, H=data.temporary_data.K,
	                    NN_mat=data.const_struct.NN_mat[K], V_matrix=..., N=..., phi=..., density_sign=...) -> H

	Fill the 2N×2N BdG matrix in the basis (c_1↑ … c_N↑, c†_1↓ … c†_N↓) into `H`:
	- hopping -t·`NN_mat` (with boundary twists and Peierls phases) in the up block and
	  its negative conjugate in the down block;
	- diagonal ∓(μ - V_i + cos(Q·r_i) φ_i);
	- s-wave pairing -Δ_i on the (i, i+N) entries; d-wave bond pairing -2Δ_dx,i, -2Δ_dy,i
	  on the bonds (experimental; the factor 2 is not settled).
	The keywords let the same function build the travelling-cluster matrix (cluster
	fields, cluster `NN_mat`, `N` = cluster size). Non-s-wave needs `SC_L = 1`.
	"""
	function make_H_matrix_K(data::LatticeData, Delta_s::Array{ComplexF64,1}, Delta_dx::Array{ComplexF64,1}, Delta_dy::Array{ComplexF64,1}; K=1, H=data.temporary_data.K, NN_mat=data.const_struct.NN_mat[K], V_matrix=data.const_struct.V_matrix, N=data.sim_data.N, phi=data.current_state.phi, density_sign=data.const_struct.density_sign)
		# The d-wave only works for SC_L = 1 for now
		if data.sim_data.SC_L != 1 && !(data.sim_data.pairing_symmetry == "s-wave")
			error("$(data.sim_data.pairing_symmetry) pairing symmetry is only implemented for SC_L = 1")
		end
		t = data.sim_data.t
		mu = data.sim_data.mu
		NN_x = data.const_struct.NN_x
		NN_y = data.const_struct.NN_y
		do_d_wave_terms = data.sim_data.pairing_symmetry in data.const_struct.d_wave_pairings
		do_s_wave_terms = data.sim_data.pairing_symmetry in data.const_struct.s_wave_pairings

		fill!(H, ComplexF64(0.0))

		for i in 1:N
			for j in 1:N
				if NN_mat[i,j] == 0
					continue
				end
				# hopping term
				# NN_mat holds 1 for nearest neighbors inside a supercell and exp(±ikx), exp(±iky) for hopping accross supercells
				# the peierls phases are already included in NN_mat
				h = -t * NN_mat[i,j]
				H[i,j] += h
				H[i+N,j+N] += - conj(h)

				# Delta terms - d-wave
				if do_d_wave_terms
					# x direction
					NN_x_ij = NN_x[i,j]
					if NN_x_ij == 1.0
						# dx = -Delta_dx[i]
						dx = -Delta_dx[i] * 2 # old - gave U->2U ?
						# dx = -Delta_dx[i] * 4
						H[i, j+N] += dx
						H[j, i+N] += dx
						H[i+N, j] += conj(dx)
						H[j+N, i] += conj(dx)
					end

					# y direction
					NN_y_ij = NN_y[i,j]
					if NN_y_ij == 1.0
						# dy = -Delta_dy[i]
						dy = -Delta_dy[i] * 2 # old - gave U->2U ?
						# dy = -Delta_dy[i] * 4
						H[i, j+N] += dy
						H[j, i+N] += dy
						H[i+N, j] += conj(dy)
						H[j+N, i] += conj(dy)
					end
				end
			end

			#chemical potential and disorder term
			# density-channel field enters as a local chemical-potential shift:
			# mu -> mu + density_sign[i]*phi[i]. For phi=0 this is the legacy diagonal.
			mu_V = mu - V_matrix[i] + density_sign[i]*phi[i]
			H[i,i] = -mu_V
			H[i+N,i+N] = mu_V
			#Delta term -s-wave
			if do_s_wave_terms
				H[i,i+N] += -Delta_s[i]
				H[i+N,i] += -conj(Delta_s[i])
			end
		end

		return H
	end



	"""
	    MCMF_action_K(data, Delta_s, Delta_dx, Delta_dy) -> (S, Energies, U_matrix)

	Diagonalize the full lattice for every supercell wavevector K and return the action

	    S = -2 Σ_K Σ_{n=1}^{N} ln cosh(β E_n/2)
	        + β SC_N Σ_i [ |Δ_i|²/(α U_i) + 2(|Δ_dx,i|²/U_x,i + |Δ_dy,i|²/U_y,i)
	                       + φ_i²/(2 β_φ U_i) - cos(Q·r_i) φ_i ]

	(sites with U = 0 contribute no field terms), with the eigenvalues (2N × SC_N) and
	eigenvectors (2N × 2N × SC_N), kept in `temporary_data.*_Bckup`. For `SC_L = 1`
	this is the single K = 0 sector. ln cosh x is replaced by x - ln 2 for x ≥ 500.
	"""
	function MCMF_action_K(data::LatticeData, Delta_s::Array{ComplexF64,1}, Delta_dx::Array{ComplexF64,1}, Delta_dy::Array{ComplexF64,1})
		sim_data = data.sim_data
		beta = sim_data.beta
		U_params = data.const_struct.U_params
		U_params_x = data.const_struct.U_params_x
		U_params_y = data.const_struct.U_params_y
		SC_N = sim_data.SC_L^2
		N = sim_data.N

		for K_ix in 1:SC_N
			# diagonalize each K separately
			data.temporary_data.K = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; H=data.temporary_data.K, NN_mat=data.const_struct.NN_mat[K_ix], V_matrix=data.const_struct.V_matrix, N=N)
			data.temporary_data.Energies_K, data.temporary_data.U_matrix_K = FastLapackInterface.syevr!(data.temporary_data.ws, 'V', 'A', 'U', data.temporary_data.K, 1.0, 1.0, 1, 1, 10e-10; resize=false)
			data.temporary_data.Energies_Bckup[:, K_ix] .= data.temporary_data.Energies_K
			data.temporary_data.U_matrix_Bckup[:, :, K_ix] .= data.temporary_data.U_matrix_K
		end

		S = 0.0
		log2 = log(2)
		for K_ix in 1:SC_N
			# compute the action summing over all K points
			for i in 1:N
				# the N lowest eigenvalues (the BdG spectrum is ±E)
				bE2 = abs(beta * data.temporary_data.Energies_Bckup[i, K_ix] / 2)
				add_term = 0.0
				if bE2 < 500
					add_term = -2*log(cosh(bE2))
				else
					add_term = -2*(bE2 - log2)
				end
				S += add_term
			end
		end

		# Pairing penalty is scaled by 1/alpha (alpha = fraction of U in the pairing
		# channel). The density-channel field phi adds phi^2/(2*beta_phi*U). alpha=1,
		# beta_phi=0 reproduces the legacy action exactly.
		alpha = sim_data.alpha
		beta_phi = sim_data.beta_phi
		phi = data.current_state.phi
		density_sign = data.const_struct.density_sign
		Srs = 0.0
		Srx = 0.0
		Sry = 0.0
		Sphi = 0.0
		# Slin: the -sum_i e^{iQ.r_i} phi_i c-number from writing K in the
		# particle-hole-symmetric BdG form (dropped when only the BdG matrix is
		# diagonalized). Included so the action is the true grand potential and the
		# phi saddle is the physical beta_phi*U*cos(Q.r)*n_i. Zero when phi=0.
		Slin = 0.0
		for i in 1:N
			if U_params[i] != 0.0
				@inbounds Srs += abs2(Delta_s[i]) / (alpha * U_params[i])
				if beta_phi != 0.0
					@inbounds Sphi += phi[i]^2 / (2 * beta_phi * U_params[i])
					@inbounds Slin += density_sign[i] * phi[i]
				end
			end
			if U_params_x[i] != 0.0
				@inbounds Srx += abs2(Delta_dx[i]) / U_params_x[i]
			end
			if U_params_y[i] != 0.0
				@inbounds Sry += abs2(Delta_dy[i]) / U_params_y[i]
			end
		end
		S += beta * SC_N * (Srs + 2*(Srx + Sry) + Sphi - Slin)

		return S, data.temporary_data.Energies_Bckup, data.temporary_data.U_matrix_Bckup
	end


	"""
	    MCMF_action_TCluster_K0(data, i) -> (S_cluster, cluster_energies, cluster_U_matrix)

	Travelling cluster approximation (K0-TCA): the action of the `TClustersize`² cluster
	centred on site `i` (same formula as `MCMF_action_K0`, restricted to the cluster
	sites, with periodic boundaries on the cluster). Only eigenvalues are computed; the
	returned eigenvector matrix is not updated. Used for the ΔS of a single-site update.
	Note: the d-wave bond term has factor 1 here and 2 in the full-lattice actions
	(d-wave is experimental).
	"""
	function MCMF_action_TCluster_K0(data::LatticeData, i::Int64)
		assignments =data.const_struct.TCluster_assignments[i]
		TCluster_Delta_s = data.temporary_data.TCluster_Delta_s
		TCluster_Delta_dx = data.temporary_data.TCluster_Delta_dx
		TCluster_Delta_dy = data.temporary_data.TCluster_Delta_dy
		TCluster_V = data.temporary_data.TCluster_V
		TCluster_phi = data.temporary_data.TCluster_phi
		TCluster_dsign = data.temporary_data.TCluster_dsign
		Delta_s = data.current_state.Delta_s
		Delta_dx = data.current_state.Delta_dx
		Delta_dy = data.current_state.Delta_dy
		phi = data.current_state.phi
		density_sign = data.const_struct.density_sign
		V_matrix = data.const_struct.V_matrix
		U_params_s = data.const_struct.U_params
		U_params_x = data.const_struct.U_params_x
		U_params_y = data.const_struct.U_params_y
		alpha = data.sim_data.alpha
		beta_phi = data.sim_data.beta_phi

		# Gather TCluster local fields without indexed-copy allocations.
		@inbounds for j in eachindex(assignments)
			ix = assignments[j]
			TCluster_Delta_s[j] = Delta_s[ix]
			TCluster_Delta_dx[j] = Delta_dx[ix]
			TCluster_Delta_dy[j] = Delta_dy[ix]
			TCluster_V[j] = V_matrix[ix]
			TCluster_phi[j] = phi[ix]
			TCluster_dsign[j] = density_sign[ix]
		end

		beta = data.sim_data.beta
		S = 0.0
		Srs = 0.0
		Srx = 0.0
		Sry = 0.0
		Sphi = 0.0
		Slin = 0.0   # -sum_i e^{iQ.r_i} phi_i c-number (see MCMF_action_K)

		# make the K matrix
		H = data.temporary_data.TCluster_K
		TCluster_N = data.sim_data.TCluster_size_x * data.sim_data.TCluster_size_y
		H = make_H_matrix_K(data, TCluster_Delta_s, TCluster_Delta_dx, TCluster_Delta_dy;
							H=H, NN_mat=data.const_struct.TCluster_NN_mats[1], V_matrix=TCluster_V, N=TCluster_N,
							phi=TCluster_phi, density_sign=TCluster_dsign)
		
		# TCA uses only eigenvalues for the local action update.
		data.temporary_data.TCluster_energies, _ = FastLapackInterface.syevr!(data.temporary_data.TCluster_ws, 'N', 'A', 'U', H, 1.0, 1.0, 1, 1, 10e-10; resize=false)

		ln2 = log(2)
		for i in 1:TCluster_N
			# compute the action in the cluster
			bE2 = abs(beta * data.temporary_data.TCluster_energies[i] / 2)
			if bE2 < 20
				@inbounds S += -2*log(cosh(bE2))
			else
				@inbounds S += -2*(bE2 - ln2)
			end
			
			# field contributions
			ix = assignments[i]
			if U_params_s[ix] != 0.0
				Srs += abs2(TCluster_Delta_s[i]) / (alpha * U_params_s[ix])
			end
			if U_params_x[ix] != 0.0
				Srx += abs2(TCluster_Delta_dx[i]) / U_params_x[ix]
			end
			if U_params_y[ix] != 0.0
				Sry += abs2(TCluster_Delta_dy[i]) / U_params_y[ix]
			end
			if beta_phi != 0.0 && U_params_s[ix] != 0.0
				Sphi += TCluster_phi[i]^2 / (2 * beta_phi * U_params_s[ix])
				Slin += TCluster_dsign[i] * TCluster_phi[i]
			end
		end

		S += beta * (Srs + 1*(Srx + Sry) + Sphi - Slin)

		return S, data.temporary_data.TCluster_energies, data.temporary_data.TCluster_U_matrix
	end


	"""
	    MCMF_action_K0(data, Delta_s, Delta_dx, Delta_dy) -> (S_K0, Energies, U_matrix)

	Action of the K = 0 sector only (the `"K0"` method): diagonalize the full 2N×2N BdG
	matrix once and return S as in `MCMF_action_K` (for d-wave symmetries only the bond
	terms, otherwise only the on-site term), with the eigenvalues and eigenvectors.
	ln cosh x is replaced by x - ln 2 for x ≥ 20.
	"""
	function MCMF_action_K0(data::LatticeData, Delta_s::Array{ComplexF64,1}, Delta_dx::Array{ComplexF64,1}, Delta_dy::Array{ComplexF64,1})
		sim_data = data.sim_data
		beta = sim_data.beta
		U_params = data.const_struct.U_params
		U_params_x = data.const_struct.U_params_x
		U_params_y = data.const_struct.U_params_y
		N = sim_data.N
		
		data.temporary_data.K = make_H_matrix_K(data, Delta_s, Delta_dx, Delta_dy; H=data.temporary_data.K, NN_mat=data.const_struct.NN_mat[1], V_matrix=data.const_struct.V_matrix, N=N)
		data.temporary_data.Energies_K, data.temporary_data.U_matrix_K = FastLapackInterface.syevr!(data.temporary_data.ws, 'V', 'A', 'U', data.temporary_data.K, 1.0, 1.0, 1, 1, 10e-10; resize=false)

		
		# compute the action
		# alpha scales the pairing penalty (1/alpha), beta_phi>0 adds the density
		# channel penalty phi^2/(2*beta_phi*U). alpha=1, beta_phi=0 => legacy action.
		alpha = sim_data.alpha
		beta_phi = sim_data.beta_phi
		phi = data.current_state.phi
		density_sign = data.const_struct.density_sign
		S = 0.0
		ln2 = log(2)
		Srs = 0.0
		Srx = 0.0
		Sry = 0.0
		Sphi = 0.0
		Slin = 0.0   # -sum_i e^{iQ.r_i} phi_i c-number (see MCMF_action_K)
		for i in 1:N
			# compute the action summing over all supercells
			bE2 = abs(beta * data.temporary_data.Energies_K[i] / 2)
			if bE2 < 20
				@inbounds S += -2*log(cosh(bE2))
			else
				@inbounds S += -2*(bE2 - ln2)
			end

			if sim_data.pairing_symmetry in data.const_struct.d_wave_pairings
				# d-wave or similar
				if U_params_x[i] != 0.0
					Srx +=  abs2(Delta_dx[i]) / U_params_x[i]
				end
				if U_params_y[i] != 0.0
					Sry +=  abs2(Delta_dy[i]) / U_params_y[i]
				end
			else
				# s-wave or similar # will not work for s+d
				if U_params[i] != 0.0
					Srs +=  abs2(Delta_s[i]) / (alpha * U_params[i])
				end
			end

			# density channel: on-site U_params, independent of pairing symmetry.
			if beta_phi != 0.0 && U_params[i] != 0.0
				@inbounds Sphi += phi[i]^2 / (2 * beta_phi * U_params[i])
				@inbounds Slin += density_sign[i] * phi[i]
			end
		end
		S += beta * (Srs + 2*(Srx + Sry) + Sphi - Slin) # try

		return S, data.temporary_data.Energies_K, data.temporary_data.U_matrix_K
	end

end

begin # Saving helper functions
	# Files of one seed folder:
	#   checkpoint.jld2       sim_data + current_state (rewritten at every save)
	#   results.jld2          sim_data + measurement_1, measurement_2, ... (one chunk per dump)
	#   field_snapshots.jld2  raw field snapshots (optional)

	"""
	    is_empty_ss_chunk(ss) -> Bool

	True if a chunk's action stream is empty or all zero (an unfilled buffer).
	"""
	function is_empty_ss_chunk(ss)
		return isempty(ss) || all(x -> x == 0.0, ss)
	end

	"""
	    count_dumped_measurements(results_path) -> Int

	Number of measurements stored in results.jld2 (non-empty chunks only); 0 if the file
	is missing or unreadable.
	"""
	function count_dumped_measurements(results_path::String)
		if !isfile(results_path)
			return 0
		end

		total_dumped = 0
		try
			jldopen(results_path, "r") do file
				meas_keys = filter(k -> startswith(k, "measurement_"), keys(file))
				meas_keys_sorted = sort(meas_keys, by=k -> parse(Int, split(k, "_")[2]))
				for k in meas_keys_sorted
					chunk = file[k]
					if !haskey(chunk, "Ss")
						continue
					end
					ss = chunk["Ss"]
					if is_empty_ss_chunk(ss)
						continue
					end
					total_dumped += length(ss)
				end
			end
		catch e
			@warn "Failed to count dumped measurements from $results_path: $e"
			return 0
		end

		return total_dumped
	end

	"""
	    reconcile_measurement_counters(num_current, saving_ix, dumped_count) -> (num_current, saving_ix)

	Make a checkpoint's measurement counters agree with results.jld2. The measurement
	buffer is not checkpointed, so measurements not yet dumped are lost on a restart:
	if the checkpoint has pending ones, the count is set to what is on disk and the
	buffer index to 0. The count never goes below what is on disk.
	"""
	function reconcile_measurement_counters(num_current::Int64, saving_ix::Int64, dumped_count::Int64)
		new_num_current = num_current
		new_saving_ix = saving_ix

		if dumped_count < 0
			dumped_count = 0
		end

		# If checkpoint says there are pending in-memory measurements, but we do not
		# persist the measurement buffer in checkpoint, trust results.jld2 as source of truth.
		if new_saving_ix > 0
			if dumped_count < new_num_current
				new_num_current = dumped_count
			elseif dumped_count > new_num_current
				new_num_current = dumped_count
			end
			new_saving_ix = 0
		end

		# Keep counters monotonic with what already exists on disk.
		if new_num_current < dumped_count
			new_num_current = dumped_count
		end

		return new_num_current, new_saving_ix
	end

	"""
	    reconcile_checkpoint_with_results!(current_state, path)

	Apply `reconcile_measurement_counters` to a loaded checkpoint, using the
	results.jld2 in `path`; warns if the counters change.
	"""
	function reconcile_checkpoint_with_results!(current_state::CurrentState, path::String)
		results_path = joinpath(path, "results.jld2")
		dumped_count = Int64(count_dumped_measurements(results_path))

		old_num = current_state.num_current_measurments
		old_ix = current_state.saving_measurement_ix
		new_num, new_ix = reconcile_measurement_counters(old_num, old_ix, dumped_count)

		if new_num != old_num || new_ix != old_ix
			@warn "Reconciled checkpoint counters with results file" old_num old_ix dumped_count new_num new_ix
			current_state.num_current_measurments = new_num
			current_state.saving_measurement_ix = new_ix
		end
	end

	"""
	    save_current_state(data, path)

	Write the checkpoint `path/checkpoint.jld2` (sim_data and current_state, as Dicts).
	"""
	function save_current_state(data::LatticeData, path::String)
		checkpoint_path = joinpath(path, "checkpoint.jld2")
		jldopen(checkpoint_path, "w") do file
			file["sim_data"] = struct2dict(data.sim_data)
			file["current_state"] = struct2dict(data.current_state)
		end
	end


	"""
	    dump_measurements(data, path)

	Save progress: write the checkpoint, append the measurements in the buffer as a new
	chunk `measurement_<k>` of `path/results.jld2` (creating the file with sim_data if
	needed), empty the buffer, and write the checkpoint again.
	"""
	function dump_measurements(data::LatticeData, path::String)
		# save the current state to "checkpoint.jld2"
		save_current_state(data, path)

		# prepare the data for dumping
		dump_data = prepare_dump(data)
		if dump_data === nothing
			# Ensure any internal counter resets done in prepare_dump are persisted.
			save_current_state(data, path)
			return
		end

		# dump the measurments to "results.jld2"
		results_path = joinpath(path, "results.jld2")
		
		if isfile(results_path)
			# dump measurments to a new chunk
			jldopen(results_path, "r+") do file
				existing_chunks = filter(x -> startswith(x, "measurement_"), keys(file))
				taken_ids = map(x -> parse(Int, split(x, "_")[2]), existing_chunks)
				next_chunk_id = isempty(taken_ids) ? 1 : maximum(taken_ids) + 1
				
				file["measurement_$(next_chunk_id)"] = dump_data
			end
		else
			# if the file does not exist, create it and save the first chunk
			jldopen(results_path, "w") do file
				file["sim_data"] = struct2dict(data.sim_data)
				file["measurement_1"] = dump_data
			end
		end

		# Save checkpoint again after successful dump so resumed runs see the
		# reset in-memory buffer counters (saving_measurement_ix=0).
		save_current_state(data, path)
	end


	"""
	    save_field_snapshot(data, path)

	Append a snapshot of the raw MC fields (Δ_s, Δ_dx, Δ_dy, φ, plus S and the step) to
	`path/field_snapshots.jld2`, under `snapshot_<measurement index>`.

	A separate file, so it cannot interfere with results.jld2 or the checkpoint. If a
	resume reaches an index that already has a snapshot, the existing one is kept.
	"""
	function save_field_snapshot(data::LatticeData, path::String)
		n =data.current_state.num_current_measurments
		snapshot_path = joinpath(path, "field_snapshots.jld2")
		key = "snapshot_$(n)"

		snap = Dict{String, Any}(
			"num_current_measurments" => n,
			"step_count"              => data.current_state.step_count,
			"S"                       => data.current_state.S,
			"phi"                     => copy(data.current_state.phi),
			"Delta_s"                 => copy(data.current_state.Delta_s),
			"Delta_dx"                => copy(data.current_state.Delta_dx),
			"Delta_dy"                => copy(data.current_state.Delta_dy),
		)

		if isfile(snapshot_path)
			jldopen(snapshot_path, "r+") do file
				if haskey(file, key)
					return  # already snapshotted this measurement index; do not duplicate
				end
				file[key] = snap
			end
		else
			jldopen(snapshot_path, "w") do file
				# Record sim_data once so a snapshots file is self-describing.
				file["sim_data"] = struct2dict(data.sim_data)
				file[key] = snap
			end
		end
	end


	"""
	    prepare_dump(data) -> Union{Dict, Nothing}

	The filled part of the measurement buffer as a Dict (streams cut to
	`saving_measurement_ix`; running means and switched-off observables stored whole),
	then reset the buffer. Returns `nothing` if the buffer is empty or all zero.
	"""
	function prepare_dump(data::LatticeData)
		num_dumped =data.current_state.saving_measurement_ix
		if num_dumped == 0
			return nothing
		end

		# Guard against stale counters after interrupted runs: if the pending slice is
		# entirely zero-initialized, drop it instead of writing an empty chunk.
		ss_pending = data.measurement_data.Ss[1:num_dumped]
		if is_empty_ss_chunk(ss_pending)
			@warn "Dropping zero-only pending measurement buffer" num_dumped num_current=data.current_state.num_current_measurments
			data.measurement_data = Initial_measurments(data.sim_data)
			data.current_state.saving_measurement_ix = 0
			return nothing
		end

		obs_no_cut = [:BZ, :A, :LDOS]
		measure_dwave_observables = data.sim_data.pairing_symmetry in ["d-wave", "d-p-wave", "mayr-d-wave", "d+s-wave"]
		if !measure_dwave_observables
			append!(obs_no_cut, [:Phase_Corr_dwave_x, :Phase_Corr_dwave_y, :Phase_Corr_mixed])

			# Non-d-wave modes initialize these observables as empty vectors; keep
			# dump robust if old/foreign checkpoint state leaked non-empty buffers.
			for obs in (:Phase_Corr_dwave_x, :Phase_Corr_dwave_y, :Phase_Corr_mixed)
				obs_data = getfield(data.measurement_data, obs)
				if !isempty(obs_data)
					@warn "Non-empty d-wave phase observable in non-d-wave mode; storing full field without slicing" pairing_symmetry=data.sim_data.pairing_symmetry obs length=length(obs_data)
				end
			end
		end
		# When jP is off, its fields are empty: dump whole (no slicing) like the d-wave fields.
		if !data.sim_data.compute_jP
			append!(obs_no_cut, [:jPx_q0, :jPx_qy, :jPx_qx, :jPy_q0, :jPy_qy, :jPy_qx])
		end
		dump_data = Dict{String, Any}()
		
		for obs in fieldnames(MeasureData)
			if obs in obs_no_cut
				dump_data[string(obs)] = getfield(data.measurement_data, obs)
				continue
			end
			dump_data[string(obs)] = getfield(data.measurement_data, obs)[1:num_dumped]
		end

		data.measurement_data = Initial_measurments(data.sim_data) # reset the measurement data
		data.current_state.saving_measurement_ix = 0 # reset the saving_measurement_ix

		return dump_data
	end


	"""
	    assert_not_slimmed(dir_path)

	Stop with an error if the seed folder was slimmed (results.jld2 replaced by
	results_slim.jld2). Its checkpoint survives, but the measurement stream its counters
	refer to does not, so such a seed can never be continued.
	"""
	function assert_not_slimmed(dir_path::String)
		if isfile(joinpath(dir_path, "results_slim.jld2")) && !isfile(joinpath(dir_path, "results.jld2"))
			error("$dir_path was slimmed (results.jld2 replaced by results_slim.jld2) and cannot be continued. Run it as a new unit in a fresh folder.")
		end
	end

	"""
	    load_checkpoint(path; cluster="") -> Union{LatticeData, Nothing}

	Rebuild a simulation from `path/checkpoint.jld2`: sim_data and current_state from the
	file, constants and scratch rebuilt, empty measurement buffer, counters reconciled
	with results.jld2. `path` goes through `fix_cluster_path`. Returns `nothing` (with an
	error message) if the file is missing or unreadable.
	"""
	function load_checkpoint(path::String; cluster::AbstractString="")
		path =fix_cluster_path(path; cluster=cluster)
		checkpoint_path = joinpath(path, "checkpoint.jld2")
		if !isfile(checkpoint_path)
			@error "Checkpoint file not found: $checkpoint_path"
			return nothing
		end

		try
			sim_data, current_state = jldopen(checkpoint_path, "r") do file
				dict2struct(file["sim_data"]), dict2struct(file["current_state"])
			end
			data = LatticeData(sim_data, current_state, Initial_measurments(sim_data), Const_Struct(sim_data), Temporary_Data(sim_data))
			reconcile_checkpoint_with_results!(data.current_state, path)
			return data
		catch e
			@error "Failed to load checkpoint from $checkpoint_path: $e"
			return nothing
		end
	end

	"""
	    load_results(path; to_load_checkpoint=true, cluster="") -> Union{LatticeData, Nothing}

	Load all measurements of one seed folder (`path` goes through `fix_cluster_path`).

	The chunks of results.jld2 are joined into one `MeasureData`: streams are
	concatenated, the running means (A, BZ, LDOS) are averaged weighted by chunk size.
	Empty chunks and observables missing from some chunks are skipped (with a warning);
	old `drude_dc_*` keys are renamed `L_D_dc_*`. With `to_load_checkpoint` the state comes
	from the checkpoint, otherwise a fresh initial state is used and
	`num_current_measurments` is set to the number of measurements loaded.
	Returns `nothing` if results.jld2 is missing or unreadable.
	"""
	function load_results(path::String; to_load_checkpoint=true, cluster::AbstractString="")
		path = fix_cluster_path(path; cluster=cluster)

		results_path = joinpath(path, "results.jld2")
		if !isfile(results_path)
			@warn "Results file not found: $results_path"
			return nothing
		end
		mean_obs = ["BZ", "A", "LDOS"]
		measurments = nothing
		sim_data = nothing

		# load measurments and sim_data
		try
			(measurments, sim_data) = with_nan_fallback_warn_context(results_path) do
				jldopen(results_path, "r") do file
				chunks = []
				chunk_sizes = []
				drude_key_aliases = [
					("drude_dc_xx", "L_D_dc_xx"),
					("drude_dc_xy", "L_D_dc_xy"),
					("drude_dc_yx", "L_D_dc_yx"),
					("drude_dc_yy", "L_D_dc_yy"),
				]
				meas_keys = filter(k -> startswith(k, "measurement_"), keys(file))
				meas_keys_sorted = sort(meas_keys, by=k -> parse(Int, split(k, "_")[2]))
				for k in meas_keys_sorted
					raw_chunk = file[k]
					chunk = Dict{String, Any}()
					for obs in keys(raw_chunk)
						chunk[string(obs)] = raw_chunk[obs]
					end

					# Migration per chunk: rename old drude_dc_* keys -> L_D_dc_*
					for (old_key, new_key) in drude_key_aliases
						if haskey(chunk, old_key) && !haskey(chunk, new_key)
							chunk[new_key] = chunk[old_key]
							delete!(chunk, old_key)
						end
					end

					if !haskey(chunk, "Ss")
						@warn "Skipping malformed measurement chunk (missing Ss): $k"
						continue
					end
					if is_empty_ss_chunk(chunk["Ss"])	#skip empty chunks
						@warn "Skipping empty measurement chunk: $k"
						continue
					end
					push!(chunks, chunk)
					push!(chunk_sizes, length(chunk["Ss"]))
				end

				if isempty(chunks)
					error("No non-empty measurement chunks found in $results_path")
				end

				# Concatenate the chunks
				shared_obs = Set(keys(chunks[1]))
				all_obs = Set(keys(chunks[1]))
				for chunk in chunks[2:end]
					shared_obs = intersect(shared_obs, Set(keys(chunk)))
					union!(all_obs, Set(keys(chunk)))
				end
				dropped_obs = sort(collect(setdiff(all_obs, shared_obs)))
				if !isempty(dropped_obs)
					@warn "Skipping observables missing in some chunks: $(join(dropped_obs, ", "))"
				end
				obs_names = sort(collect(shared_obs))
				obs_vcat = Dict{String, Any}("type"=>:MeasureData)
				for obs in obs_names
					if obs == "type"
						continue
					end
					if obs in mean_obs
						obs_vcat[obs] = sum([chunk[obs]* chunk_sizes[i] for (i, chunk) in enumerate(chunks)])/ sum(chunk_sizes)
					else
						obs_vcat[obs] = vcat([chunk[obs] for chunk in chunks]...)
					end
				end
				# Migration: rename old drude_dc_* keys → L_D_dc_*
				for (old_key, new_key) in drude_key_aliases
					if haskey(obs_vcat, old_key) && !haskey(obs_vcat, new_key)
						obs_vcat[new_key] = obs_vcat[old_key]
						delete!(obs_vcat, old_key)
					end
				end

					(dict2struct(obs_vcat), dict2struct(file["sim_data"]))
				end
			end
		catch e
			@error "Failed to load results from $results_path: $e"
			return nothing
		end

		if to_load_checkpoint
			data = load_checkpoint(path)
			if !isnothing(data)
				data.measurement_data = measurments
				return data
			end
		end

		const_data = Const_Struct(sim_data)
		data = LatticeData(sim_data, Initial_state(sim_data; const_data=const_data), measurments, const_data, Temporary_Data(sim_data))
		data.current_state.num_current_measurments = length(data.measurement_data.Ss)
		return data
	end



    """
        struct2dict(s) -> Dict{String,Any}

    A simulation struct as a Dict (field name => value, nested structs converted too),
    with `"type"` = the struct name. This is how structs are saved to JLD2, so files stay
    readable when a struct gains fields. Inverse: `dict2struct`.
    """
    function struct2dict(s::Any)
        d = Dict{String, Any}()
        struct_type = typeof(s)
		struct_names = [ConstStruct, SimData, MeasureData, LatticeData, TemporaryData, CurrentState]
        d["type"] = string(struct_type)
        for field in fieldnames(typeof(s))
            value = getfield(s, field)
            if typeof(value) in struct_names
                d[string(field)] = struct2dict(value)
            else 
                d[string(field)] = value

            end
        end
        return d
    end

	# Loading old files: a MeasureData field whose stored type does not match the
	# current struct is replaced by NaNs of the right shape (nan_field_fallback), with
	# one warning per (file, field).
	nan_fallback_warn_context = Ref("")
	nan_fallback_warned_fields = Set{Tuple{String, Symbol}}()

	"""
	    with_nan_fallback_warn_context(f, results_path)
	    with_nan_fallback_warn_context(results_path, f)

	Run `f()` with `results_path` recorded as the file named in NaN-fallback warnings.
	"""
	function with_nan_fallback_warn_context(f::Function, results_path::String)
		prev = nan_fallback_warn_context[]
		nan_fallback_warn_context[] = results_path
		try
			return f()
		finally
			nan_fallback_warn_context[] = prev
		end
	end

	function with_nan_fallback_warn_context(results_path::String, f::Function)
		return with_nan_fallback_warn_context(f, results_path)
	end

	"""
	    warn_nan_fallback_once(field, expected_type, got_type)

	Warn (once per file and field) that a stored field had the wrong type and was replaced by NaNs.
	"""
	function warn_nan_fallback_once(field::Symbol, expected_type, got_type)
		results_path = isempty(nan_fallback_warn_context[]) ? "<unknown>" : nan_fallback_warn_context[]
		key = (results_path, field)
		if key in nan_fallback_warned_fields
			return
		end
		push!(nan_fallback_warned_fields, key)
		@warn "MeasureData field type mismatch; using NaN fallback" field expected_type got_type results_path
	end

	"""
	    default_struct_field_value(ftype)

	Value for a struct field missing from an old file: 0, "", an empty array of the
	right type, or `nothing`.
	"""
	function default_struct_field_value(ftype::Type)
		if ftype <: Number
			return zero(ftype)
		elseif ftype <: AbstractString
			return ""
		elseif ftype <: AbstractArray
			nd = ndims(ftype)
			return Array{eltype(ftype), nd}(undef, ntuple(_ -> 0, nd)...)
		end
		return nothing
	end

	"""
	    nan_scalar_fallback(ftype)

	NaN of a float or complex type (NaN + NaN·i); zero for other number types.
	"""
	function nan_scalar_fallback(ftype::Type)
		if ftype <: AbstractFloat
			return ftype(NaN)
		elseif ftype <: Complex
			real_type = ftype.parameters[1]
			if real_type <: AbstractFloat
				return ftype(real_type(NaN), real_type(NaN))
			end
		end
		return zero(ftype)
	end

	"""
	    nan_array_element_fallback(el_t, template)

	NaN replacement for one array element of type `el_t`, shaped like `template`.
	"""
	function nan_array_element_fallback(el_t::Type, template)
		if el_t <: AbstractArray
			return nan_array_fallback(el_t, template)
		elseif el_t <: Number
			return nan_scalar_fallback(el_t)
		end
		fallback = default_struct_field_value(el_t)
		return isnothing(fallback) ? template : fallback
	end

	"""
	    nan_array_fallback(ftype, template)

	Array of type `ftype` with the shape of `template`, filled with NaNs (nested arrays
	too); an empty array if `template` is not an array of the same dimension.
	"""
	function nan_array_fallback(ftype::Type, template)
		nd = ndims(ftype)
		el_t = eltype(ftype)
		if !(template isa AbstractArray)
			return default_struct_field_value(ftype)
		end

		if nd == 1
			return [nan_array_element_fallback(el_t, entry) for entry in template]
		end

		if ndims(template) == nd
			output = Array{el_t, nd}(undef, size(template)...)
			for idx in CartesianIndices(template)
				output[idx] = nan_array_element_fallback(el_t, template[idx])
			end
			return output
		end

		return default_struct_field_value(ftype)
	end

	"""
	    nan_field_fallback(ftype, template)

	NaN replacement for a whole field of type `ftype`, shaped like the stored `template`.
	"""
	function nan_field_fallback(ftype::Type, template)
		if ftype <: AbstractArray
			return nan_array_fallback(ftype, template)
		elseif ftype <: Number
			return nan_scalar_fallback(ftype)
		end
		fallback = default_struct_field_value(ftype)
		return isnothing(fallback) ? template : fallback
	end

    """
        dict2struct(d::Dict{String,Any})

    Rebuild the struct named by `d["type"]` from a Dict written by `struct2dict`
    (nested Dicts too). Handles old files: missing fields get defaults (`phi` zeros of
    the lattice size, `diag_is_synced = true`), the old per-step layout of `A` is averaged
    into the new one, and MeasureData fields of the wrong type become NaNs.
    """
    function dict2struct(d::Dict{String, Any})
        for (k, v) in d
            if isa(v, Dict)
                d[k] = dict2struct(v)
            end
        end
        struct_type = eval(Symbol(d["type"]))
        fields = fieldnames(struct_type)
        values = Any[]
        for field in fields
			ftype = fieldtype(struct_type, field)
            if !haskey(d, string(field))
				if struct_type == CurrentState && field == :diag_is_synced
					# Legacy checkpoints were written from always-synced states.
					push!(values, true)
				elseif struct_type == CurrentState && field == :diag_synced_step
					push!(values, haskey(d, "step_count") ? Int64(d["step_count"]) : Int64(0))
				elseif struct_type == CurrentState && field == :phi
					# Legacy checkpoints predate the density channel: default phi to
					# zeros of the right length (inferred from Delta_s), not length 0.
					n_phi = haskey(d, "Delta_s") ? length(d["Delta_s"]) : 0
					push!(values, zeros(Float64, n_phi))
				else
					push!(values, default_struct_field_value(ftype))
				end

			elseif (typeof(d[string(field)]) !== ftype) && (field == :A) 
				# If the type of the value in the dictionary does not match the field type, convert it

				if (ftype ==  Array{Array{Float64, 1}, 1}) && (typeof(d[string(field)]) == Array{Array{Array{Float64, 1}, 1}, 1})
					# old structure of A
					# {{{A(k,y)} for w} for step} --> {A(k,y) for w}
					# convert to new structure
					old_A = d[string(field)]
					push!(values, [mean([old_A[ix][w] for ix in eachindex(old_A)]) for w in eachindex(old_A[1])])
				else
					push!(values, d[string(field)])
				end

			elseif (typeof(d[string(field)]) !== ftype) && (struct_type == MeasureData)
				raw_value = d[string(field)]
				warn_nan_fallback_once(field, ftype, typeof(raw_value))
				push!(values, nan_field_fallback(ftype, raw_value))
            else
                push!(values, d[string(field)])
            end
            
        end
        
        return struct_type(values...)
    end



	# ------------------------------------------------------------------------
	# Cluster paths seen from another machine.
	#
	# Every stored path (a seed's dir_path, ...) is a CLUSTER path such as
	#   /home/data/<user>/data/MC_Hubbard_V7/data/<campaign>/<unit>/seed_1
	# and both clusters use the SAME path strings, so the path alone does not say
	# which cluster's disk it is on. The part from `data/MC_Hubbard_Vx/` on is the
	# same everywhere. From another machine it is reached through (in this order):
	#   * a folder set per cluster: <root>/MC_Hubbard_V7/...        (LANDAU_ROOT / CURIE_ROOT)
	#   * the backup drive:     <drive>/<cluster>/MC_Hubbard_V7/...  (BACKUP_DRIVE)
	#   * that cluster's mount: <mount>/data/MC_Hubbard_V7/...      (LANDAU_MOUNT / CURIE_MOUNT,
	#                           a mount of the cluster's /home/data/<user>)
	# All are environment variables read when this file is loaded, with no
	# defaults: an unset one is not used. The same order as cluster_root in
	# means_db.jl. CLUSTER_DATA_PREFIX (the cluster's /home/data/<user>; default:
	# two levels above env V7_DATA_ROOT, which every job has) is needed only to map
	# a mount path back to the cluster.
	# ------------------------------------------------------------------------

	"""
	The cluster's `/home/data/<user>` (env `CLUSTER_DATA_PREFIX`; default: two levels
	above env `V7_DATA_ROOT`, else empty). Only used to map a mount path back to the
	cluster (`fix_cluster_path`); cluster paths are recognised without it.
	"""
	const CLUSTER_DATA_PREFIX = rstrip(get(ENV, "CLUSTER_DATA_PREFIX",
		isempty(get(ENV, "V7_DATA_ROOT", "")) ? "" : dirname(dirname(rstrip(ENV["V7_DATA_ROOT"], '/')))), '/')

	"""
	Where each cluster's `/home/data/<user>` is mounted on this machine (env
	`LANDAU_MOUNT`, `CURIE_MOUNT`, e.g. `Y:` and `Z:`). Empty = not used. Order
	matters: with an unknown cluster, `fix_cluster_path` tries them in this order.
	"""
	const CLUSTER_MOUNTS = [
		"landau" => String(strip(get(ENV, "LANDAU_MOUNT", ""))),
		"curie"  => String(strip(get(ENV, "CURIE_MOUNT", ""))),
	]

	"""
	The backup drive's top folder, holding `landau/` and `curie/` (env
	`BACKUP_DRIVE`). Empty = not used.
	"""
	const BACKUP_DRIVE = get(ENV, "BACKUP_DRIVE", "")

	"""
	A folder per cluster that holds its `MC_Hubbard_V6/` and `MC_Hubbard_V7/` (env
	`LANDAU_ROOT`, `CURIE_ROOT`; e.g. a cluster mount's `data` folder, or the drive's
	`<drive>/curie`). Empty = not used.
	"""
	const CLUSTER_ROOTS = Dict(name => String(strip(get(ENV, uppercase(name) * "_ROOT", "")))
	                           for name in ("landau", "curie"))

	"""
	    _join_mount(root, rel)

	`root` + "/" + `rel`, without a doubled separator.
	"""
	_join_mount(root, rel) = rstrip(root, ['/', '\\']) * "/" * rel

	"""
	    _cluster_rel(path) -> Union{String,Nothing}

	The part of a cluster path below the cluster's `/home/data/<user>`, e.g.
	`data/MC_Hubbard_V7/data/<campaign>/...`: everything after `CLUSTER_DATA_PREFIX`
	when that is set and matches, else from the first `/data/MC_Hubbard_Vx` on.
	`nothing` if `path` is not an absolute cluster (POSIX) path.
	"""
	function _cluster_rel(path::AbstractString)
		p = replace(String(path), '\\' => '/')
		startswith(p, "/") || return nothing
		if !isempty(CLUSTER_DATA_PREFIX) && startswith(p, CLUSTER_DATA_PREFIX * "/")
			return p[length(CLUSTER_DATA_PREFIX)+2:end]
		end
		m = match(r"/(data/MC_Hubbard_V\d+(?:/.*)?)$", p)
		return m === nothing ? nothing : String(m.captures[1])
	end

	"""
		cluster_path_candidates(path, cluster="") -> Vector{Pair{String,String}}

	The places on this machine where the cluster path `path` may be, as
	`cluster => local path`: the cluster's own folder first (if `LANDAU_ROOT` /
	`CURIE_ROOT` is set), then the backup drive (if `BACKUP_DRIVE` is set), then the
	cluster's mount (if `LANDAU_MOUNT` / `CURIE_MOUNT` is set). The drive comes before
	the mount because, once it is set, it is where the data is read from: a seed on
	the drive is never read over the (slow, fragile) SFTP mount, and the mount
	remains the fallback for seeds not on the drive. For `cluster = ""` the
	candidates of every cluster, in `CLUSTER_MOUNTS` order. Empty if `path` is not a
	cluster path (see `_cluster_rel`).
	"""
	function cluster_path_candidates(path::AbstractString, cluster::AbstractString="")
		rel = _cluster_rel(path)                            # e.g. "data/MC_Hubbard_V7/..."
		rel === nothing && return Pair{String,String}[]
		c = lowercase(String(cluster))
		out = Pair{String,String}[]
		for (name, mount) in CLUSTER_MOUNTS
			(isempty(c) || c == name) || continue
			root = get(CLUSTER_ROOTS, name, "")
			if !isempty(root) && startswith(rel, "data/")
				push!(out, name => _join_mount(root, rel[6:end]))
			end
			if !isempty(BACKUP_DRIVE) && startswith(rel, "data/")
				push!(out, name => _join_mount(joinpath(BACKUP_DRIVE, name), rel[6:end]))
			end
			isempty(mount) || push!(out, name => _join_mount(mount, rel))
		end
		return out
	end

	"""
		fix_cluster_path(path; cluster="") -> String

	`path` as it can be opened on THIS machine. If the folder exists, it is
	returned unchanged (always the case on the cluster itself). Otherwise a
	cluster path is mapped to `cluster`'s folder (`LANDAU_ROOT` / `CURIE_ROOT`), the
	backup drive (`BACKUP_DRIVE`) or `cluster`'s mount (see `cluster_path_candidates`),
	and a path on a mount back to `CLUSTER_DATA_PREFIX` (running on the cluster).
	Returns `path` unchanged if nothing exists.

	Pass `cluster` (`"landau"`/`"curie"`, e.g. a means_db row's `"cluster"`)
	whenever it is known: both clusters use the same path strings, so a Curie path
	mapped to Landau's mount can open a DIFFERENT simulation that happens to have
	the same campaign and unit number. Without it, every cluster is tried and a
	warning is printed if the folder exists under more than one.
	"""
	function fix_cluster_path(path::String; cluster::AbstractString="")
		isdir(path) && return path

		cands = cluster_path_candidates(path, cluster)
		if !isempty(cands)
			# Cluster known: the first existing candidate, so a folder found on the
			# drive never touches the mount (not even a stat).
			if !isempty(cluster)
				i = findfirst(cp -> isdir(last(cp)), cands)
				return i === nothing ? path : last(cands[i])
			end
			found = [c => p for (c, p) in cands if isdir(p)]
			isempty(found) && return path
			if isempty(cluster) && length(unique(first.(found))) > 1
				@warn "fix_cluster_path: $path exists on several clusters " *
					  "($(join(unique(first.(found)), ", "))); using $(first(found[1])). " *
					  "Pass `cluster=` to choose." maxlog=5
			end
			return last(found[1])
		end

		# A mount path back to the cluster prefix (running on the cluster).
		p = replace(path, '\\' => '/')
		for (_, mount) in CLUSTER_MOUNTS
			m = rstrip(replace(mount, '\\' => '/'), '/')
			if !isempty(m) && !isempty(CLUSTER_DATA_PREFIX) && startswith(lowercase(p), lowercase(m) * "/")
				changed = CLUSTER_DATA_PREFIX * p[length(m)+1:end]
				return isdir(changed) ? changed : path
			end
		end

		@warn "Not a cluster path (no /data/MC_Hubbard_Vx/) and not on a cluster mount" path maxlog=5
		return path
	end


end

begin # update suggestion
	
	"""
	    suggest_update(lattice_data; update_type="complex")

	Random Metropolis step for one field value: a complex step (real and imaginary parts
	from the 2D Gaussian `const_struct.dist`, width set by the mean U in `Const_Struct`)
	for Δ, or a real step (`update_type="real"`) for φ.
	"""
	function suggest_update(lattice_data::LatticeData; update_type::String="complex")
		rand!(lattice_data.const_struct.dist, lattice_data.temporary_data.Delta_suggestion)

		if update_type == "complex"
			return lattice_data.temporary_data.Delta_suggestion[1] + 1im*lattice_data.temporary_data.Delta_suggestion[2]
		elseif update_type == "real"
			# real step for the density field phi (reuses one component of the draw)
			return lattice_data.temporary_data.Delta_suggestion[1]
		end

	end
end




begin #saddle piont
    """
        saddle_point_delta(sim_data; d_min=0.0001, U_params=nothing)

    Clean mean-field gap at the parameters of `sim_data`, used to seed the uniform
    initial guess. U is `U_params` (its mean if a vector) or `sim_data.U`.
    s-wave: `saddle_point_delta(U, μ, β, Lx, Ly)` (returns its tuple); d-wave types: the
    k-space d-wave mean field (`running_dwave_MF`), returned as `[mean |Δ|]`.
    """
    function saddle_point_delta(sim_data::SimData; d_min=0.0001, U_params=nothing)
		if typeof(U_params) == Array{Float64,1}
			U = mean(U_params)
		elseif typeof(U_params) == Float64
			U = U_params
		else
			U = sim_data.U
		end
	    t = sim_data.t
        mu = sim_data.mu
        beta = sim_data.beta
        Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		pairing_symmetry = sim_data.pairing_symmetry
		if pairing_symmetry == "s-wave"
			return saddle_point_delta(U, mu, beta, Lx, Ly; d_min=d_min, t=t)
		elseif pairing_symmetry in ["d-wave", "mayr-d-wave", "d-p-wave"]
			kspaceMF = running_dwave_MF(Lx, Ly  # Now supports Lx != Ly
			, sim_data.t
			, sim_data.mu
			, sim_data.beta
			; Ux=U, Uy=U, max_iter=1000, tol=1e-8, a=0.6)
			return [mean(abs.(kspaceMF[1]))]
		end
        
    end

	"""
	    saddle_point_delta(U, mu, beta, Lx, Ly; d_min=0.001, t=1) -> (Δ, eq_values, index)
	    saddle_point_delta(U, mu, beta, L; ...)

	s-wave BCS gap on the finite Lx×Ly k grid: the Δ in d_min:d_min:U that best solves
	Σ_k [tanh(β E_k/2)/E_k - 2/U] = 0, with E_k = √(ξ_k² + Δ²), ξ_k = -2t(cos kx + cos ky) - μ.
	Returns Δ = 0 if the equation has no root in the range. Returns (0, 0, 0) for U < d_min.
	"""
	function saddle_point_delta(U, mu, beta, Lx, Ly; d_min=0.001, t=1)
		N = Lx * Ly
		if U < d_min
			return 0, 0, 0
		end
		delta_range= d_min:d_min:U

        ksi(kx,ky) = -2t*(cos(kx)+cos(ky)) - mu
        ek(kx,ky,delta) = (ksi(kx,ky).^2 + delta.^2)^(1/2)

		kx_list = get_kx_list(Lx)
		ky_list = get_ky_list(Ly)
        
        eq(delta) = sum([(tanh(0.5*beta*ek(kx,ky,delta))/ek(kx,ky,delta) - 2/U) for kx in kx_list for ky in ky_list])

        eq_list = [eq(delta) for delta in delta_range]

        min_ix = argmin(abs.(eq_list))
		min_delta = nothing
        
        if sign(eq_list[1]) == sign(eq_list[end]) && (min_ix == 1 || min_ix == length(eq_list))
            min_delta = 0.0
        else
            min_delta = delta_range[min_ix]
        end
        

        return min_delta, eq_list, min_ix
    end

	# Legacy wrapper for square lattice
	saddle_point_delta(U, mu, beta, L::Int; kwargs...) = saddle_point_delta(U, mu, beta, L, L; kwargs...)
end



begin #spectral function
	# A(k, ω) = (1/N) Σ_l |Σ_n e^{-ik·r_n} U_{n,l}|² L_δ(ω - E_l), with L_δ a Lorentzian of
	# width δ and U the BdG eigenvectors. The measurement uses spectral_functionV2 (or the
	# fast kernel spectral_fast!).

	"""
	    spectral_function(kx, ky, w, sigma, Lattice_data) -> ComplexF64

	A(k, ω) from the particle (`sigma = 1`) or hole (otherwise) rows of the eigenvectors,
	Lorentzian width max(w_delta, max E/N). Ignores the supercell momentum.
	Older version of `spectral_functionV2`; used only by `DOS` and `spectral_function_full`.
	"""
	function spectral_function(kx,ky,w,sigma,Lattice_data::LatticeData)
		sim_data = Lattice_data.sim_data
		N = sim_data.N
		SC_N = sim_data.SC_L^2
		# U = sim_data.U
		U_matrix = Lattice_data.current_state.U_matrix
		Energies = Lattice_data.current_state.Energies

		sigma_ix = sigma==1 ? 0 : N
		w_sign = sigma==1 ? 1 : -1
		FT_vec = view(Lattice_data.temporary_data.U_matrix_Bckup, 1, 1:N, 1) # reuse the backup matrix for the FT vector
	
		@inbounds for n in 1:N
			n_x = Lattice_data.const_struct.x[n]
			n_y = Lattice_data.const_struct.y[n]
			FT_vec[n] = exp(-1im * (kx * n_x + ky * n_y)) # with x,y in data
		end

		delt = maximum([Lattice_data.const_struct.w_delta, maximum(Energies)/(N*SC_N)])
		A = ComplexF64(0.0)
		for l in 1:2*N
			for k in 1:SC_N
				U_FT_c = 0.0 + 0.0im
				@inbounds for n in 1:N
					U_FT_c += FT_vec[n] * U_matrix[n + sigma_ix, l, k]
				end
				lorenz = (pi*delt)^-1 * (1 + delt^-2 * (Energies[l, k] - w)^2)^-1
				A += abs2(U_FT_c) * lorenz
			end
		end
		A *= N^-1

		return A

	end

	"""
	    spectral_functionV2(Lattice_data, Kix, kix, w; delt=nothing) -> Float64

	Spin-up spectral function A(k + K, ω) at lattice momentum `k_vectors[kix]` plus
	supercell momentum `SC_K_vectors[Kix]`, with Lorentzian width `delt`
	(default max(w_delta, max E/(N·SC_N))). Reference kernel for the measured A(k, ω).
	"""
	function spectral_functionV2(Lattice_data::LatticeData, Kix::Int64, kix::Int64 ,w::Float64; delt=nothing)

		sim_data = Lattice_data.sim_data
		N = sim_data.N
		SC_L = sim_data.SC_L
		SC_N = SC_L^2
		U_matrix = Lattice_data.current_state.U_matrix
		Energies = Lattice_data.current_state.Energies
		Kx, Ky = Lattice_data.const_struct.SC_K_vectors[Kix]
		kx, ky = Lattice_data.const_struct.k_vectors[kix]
		
		# make the FT vector 
		FT_vec = view(Lattice_data.temporary_data.U_matrix_Bckup, 1, 1:N, 1) # reuse the backup matrix for the FT vector
		@inbounds for n in 1:N
			n_x = Lattice_data.const_struct.x[n]
			n_y = Lattice_data.const_struct.y[n]
			FT_vec[n] = exp(-1im *((kx + Kx) * n_x + (ky + Ky) * n_y)) # with x,y in data
		end

		if delt === nothing
			delt = maximum([Lattice_data.const_struct.w_delta, maximum(Energies)/(N*SC_N)])
		end
		A = Float64(0.0)
		for l in 1:2*N
			U_FT_c = zero(ComplexF64)
			@inbounds for n in 1:N
				U_FT_c += FT_vec[n] * U_matrix[n , l, Kix]
			end
			lorenz = (pi*delt)^-1 * (1 + delt^-2 * (Energies[l, Kix] - w)^2)^-1
			A += abs2(U_FT_c) * lorenz
		end
		A *= N^-1

		return A
	end

	"""
	    spectral_function_full(w, sigma, Lattice_data)

	`spectral_function` at every k of the lattice. Not used by the pipeline.
	"""
	function spectral_function_full(w,sigma,Lattice_data::LatticeData)
		N = Lattice_data.sim_data.N
		ki_list =  Lattice_data.const_struct.ki_list
		CI = Lattice_data.const_struct.CI

		SF = Lattice_data.temporary_data.SF

		for ix in 1:N
			cur_x = Lattice_data.const_struct.x[ix]
			cur_y = Lattice_data.const_struct.y[ix]
			SF[ix] = spectral_function(ki_list[cur_x] ,ki_list[cur_y] , w, sigma, Lattice_data)
		end
		
		return SF
	end

	"""
	    spectral_function_fullV2(Lattice_data, w; delt=nothing)

	`spectral_functionV2` at every k (+K) of the (super)lattice: the BZ map A(k, ω).
	Written into `temporary_data.SF`, which is returned.
	"""
	function spectral_function_fullV2(Lattice_data::LatticeData, w::Float64; delt=nothing)
		Lx = Lattice_data.sim_data.Lx
		Ly = Lattice_data.sim_data.Ly
		L_s = Lattice_data.sim_data.SC_L
		SF = Lattice_data.temporary_data.SF

		N_gen = (Lx * L_s) * (Ly * L_s)
		for gen_ix in 1:N_gen
			k_ix, K_ix = Lattice_data.const_struct.k_general_qK_idx[gen_ix]

			SF[gen_ix] = spectral_functionV2(Lattice_data, K_ix, k_ix, w; delt=delt)
		end
		return SF
	end

	"""
	    reduced_spectral_function(lattice_data, w) -> Vector{Float64}

	Unfinished placeholder: builds a k path but returns zeros. Not used by the pipeline
	(the measured path is `ConstStruct.SF_k_path`).
	"""
	function reduced_spectral_function(lattice_data::LatticeData, w::Float64)

		sim_data = lattice_data.sim_data
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		L_min = min(Lx, Ly)
		
		# Use bounded path segments for rectangular lattices
		# Segment 1: (kx, ky=center) for kx from center to end
		seg1_kx = Int.(div(Lx,2)+1:Lx)
		seg1_ky = Int.(fill(div(Ly,2)+1, length(seg1_kx)))
		
		# Segment 2: (kx=1, ky) for ky from center+1 to end, then wrap to 1
		seg2_ky = Int.(vcat(div(Ly,2)+2:Ly, 1))
		seg2_kx = Int.(fill(1, length(seg2_ky)))
		
		# Segment 3: (kx, ky=1) for kx from end back to center
		seg3_kx = Int.(Lx:-1:div(Lx,2)+1)
		seg3_ky = Int.(fill(1, length(seg3_kx)))
		
		x_path = Int.(vcat(seg1_kx, seg2_kx, seg3_kx))
		y_path = Int.(vcat(seg1_ky, seg2_ky, seg3_ky))
		
		k_path = zip(x_path, y_path)
		k_path_ix = Int.([idx(kx,ky,Lx) for (kx,ky) in k_path])
		
		reduced_A = zeros(Float64, length(x_path))
		for (i, k) in enumerate(k_path)
			# TODO: compute spectral function at each k point
		end

		return reduced_A
	end


	"""
	    DOS(w, Lattice_data) -> ComplexF64

	Density of states at ω as Σ_k Σ_spin A(k, ω) (slow, via `spectral_function`).
	Not used by the pipeline: the measured DOS comes from the LDOS.
	"""
	function DOS(w, Lattice_data::LatticeData) ## can be comuputed from U_matrix directrly. no need for spectral function
		Lx = Lattice_data.sim_data.Lx
		Ly = Lattice_data.sim_data.Ly
		N = Lattice_data.sim_data.N
		kx_list = Lattice_data.const_struct.kx_list
		ky_list = Lattice_data.const_struct.ky_list
		CI = Lattice_data.const_struct.CI
		DOS = Float64(0)

		for i in 1:N
			for spin = [+1,-1]
				DOS += spectral_function(kx_list[get_x(i, CI)], ky_list[get_y(i, CI)], w, spin, Lattice_data)
			end
		end
		
		return DOS

	end
	
	"""
	    DOS_full(Lattice_data) -> Vector

	`DOS` on every ω of `w_list`. Not used by the pipeline.
	"""
	function DOS_full(Lattice_data::LatticeData)
		w_list = Lattice_data.sim_data.w_list
		full_dos = zeros(length(w_list))
		for (i,w) in enumerate(w_list)
			full_dos[i] = DOS(w, Lattice_data)
		end
		return full_dos

	end


end


begin #MC simulation

	"""
	    Lattice_update!(Lattice_data, Delta_s, Delta_dx, Delta_dy, S, Energies, U_matrix; compute_FD=true)

	Store new fields and their action and eigen-solution in the current state (marked as
	in sync), optionally recompute the Fermi factors and broadening, and count one step.
	"""
	function Lattice_update!(Lattice_data::LatticeData, Delta_s::Array{ComplexF64,1}, Delta_dx::Array{ComplexF64,1}, Delta_dy::Array{ComplexF64,1}, S::Float64, Energies::Array{Float64,2}, U_matrix::Array{ComplexF64,3}; compute_FD=true)

        Lattice_data.current_state.Delta_s .= Delta_s
        Lattice_data.current_state.Delta_dx .= Delta_dx
        Lattice_data.current_state.Delta_dy .= Delta_dy
        Lattice_data.current_state.S = S
        Lattice_data.current_state.Energies .= Energies
        Lattice_data.current_state.U_matrix .= U_matrix
		mark_global_diag_synced!(Lattice_data)
		if compute_FD
			# compute the Fermi-Dirac distribution for the new energies
			get_FD_dist!(Lattice_data)
			# compute the delta parameter for Lorentzian broadening
			compute_delta_analitic_continuation!(Lattice_data; method="mean_diff_low")
		end
		Lattice_data.current_state.step_count += 1
		Lattice_data.current_state.diag_synced_step = Lattice_data.current_state.step_count
    end



	"""
	    MC_stepV2!(Lattice_data; method="exact", loop_random=true, pairing="s-wave")

	One MC sweep: N single-site Metropolis updates per field (Δ_s, or Δ_dx/Δ_dy for the
	d-wave types, plus φ when β_φ ≠ 0). With `loop_random` each update picks a random site
	and field; otherwise sites are visited in order, field after field.
	`method` (`diag_method`) sets how ΔS is computed (`localized_update`): `"K0"` keeps the
	full action and eigen-solution up to date; `"K0-TCA"` only marks them out of date
	(they are recomputed before the next measurement or save). Increments `step_count`.
	"""
	function MC_stepV2!(Lattice_data::LatticeData; method="exact", loop_random=true, pairing="s-wave")
        sim_data = Lattice_data.sim_data
        N = sim_data.N
		is_exact = method == "exact"
		is_k0 = method == "K0"
		is_tcluster_like = (method == "K0-TCA") || (method == "TCluster")
		if !(is_exact || is_k0 || is_tcluster_like)
			error("unknown diag_method=\"$method\". Use \"K0\" or \"K0-TCA\" " *
				  "(\"exact\" and \"TCluster\" are the old supercell methods).")
		end

		is_s_wave = pairing == "s-wave"
		is_d_wave = pairing == "d-wave"
		is_mayr_d_wave = pairing == "mayr-d-wave"
		is_dp_wave = pairing == "d-p-wave"
		is_sd_wave = pairing == "s+d-wave"
		if !(is_s_wave || is_d_wave || is_mayr_d_wave || is_dp_wave || is_sd_wave)
			error("pairing should be either (\"s-wave\", \"d-wave\", \"mayr-d-wave\", \"d-p-wave\", \"s+d-wave\")")
		end

		if is_s_wave
			# only Delta_s is updated
			pairing_updates = [:Delta_s]
		elseif is_d_wave || is_mayr_d_wave
			# only Delta_dx and Delta_dy are updated
			pairing_updates = [:Delta_dx, :Delta_dy]
		elseif is_dp_wave
			# only Delta_dx is updated
			pairing_updates = [:Delta_dx]
		elseif is_sd_wave
			# all pairing fields are updated
			pairing_updates = [:Delta_s, :Delta_dx, :Delta_dy]
		end
		# density channel: also sweep the on-site real field phi when active.
		if sim_data.beta_phi != 0.0
			pairing_updates = vcat(pairing_updates, [:phi])
		end
		pairing_len = length(pairing_updates)
		updates_len = N * pairing_len

		# localized update
        for update_ix in 1:updates_len
			site = loop_random ? rand(1:N) : rem(update_ix - 1, N) + 1 # randomize the index
			pairing_field = loop_random ? pairing_updates[rand(1:pairing_len)] : pairing_updates[div(update_ix - 1, N) + 1] # if not random, x than y over the whole lattice
			dS, suggestion, new_energies, new_U_matrix = localized_update(Lattice_data, site, method, pairing, pairing_field) # dS=nothing if U=0
			if dS === nothing
				continue # U=0 case
			end
			accepted = check_acceptence(dS)

            if accepted
				# update the action
				if is_exact
					Lattice_data.current_state.S +=  dS
					Lattice_data.current_state.Energies .= new_energies
					Lattice_data.current_state.U_matrix .= new_U_matrix
					mark_global_diag_synced!(Lattice_data)
				elseif is_k0
					Lattice_data.current_state.S_K0 +=  dS
					if Lattice_data.sim_data.SC_L == 1
						# also update the exact action
						Lattice_data.current_state.S +=  dS
						Lattice_data.current_state.Energies .= new_energies
						Lattice_data.current_state.U_matrix .= new_U_matrix
						mark_global_diag_synced!(Lattice_data)
					else
						invalidate_global_diag!(Lattice_data)
					end
				elseif is_tcluster_like
					invalidate_global_diag!(Lattice_data)
				end


			else
				# revert the change in the pairing field
				revert_pairing_field!(Lattice_data, site, pairing, pairing_field, suggestion)
			end
		end

		Lattice_data.current_state.step_count += 1
		if Lattice_data.current_state.diag_is_synced
			Lattice_data.current_state.diag_synced_step = Lattice_data.current_state.step_count
		end
	end

	"""
	    localized_update(Lattice_data, site, method, pairing_symmetry, pairing_field; update_type="complex")
	        -> (dS, step, new_energies, new_U_matrix)

	Propose a change of `pairing_field` (`:Delta_s`, `:Delta_dx`, `:Delta_dy` or `:phi`) at
	`site`, apply it to the field, and return the change of the action ΔS (from the full
	lattice for `"K0"`, from the travelling cluster around `site` for `"K0-TCA"`), the step,
	and the new eigen-solution. The caller accepts or reverts. Returns all `nothing` if
	the site's U is 0 (no field there).
	"""
	function localized_update(Lattice_data::LatticeData, site::Int64, method::String, pairing_symmetry::String, pairing_field::Symbol; update_type::String="complex")
		# check if U is zero at the site
		if pairing_field == :Delta_s
			Us = Lattice_data.const_struct.U_params
		elseif pairing_field == :Delta_dx
			Us = Lattice_data.const_struct.U_params_x
		elseif pairing_field == :Delta_dy
			Us = Lattice_data.const_struct.U_params_y
		elseif pairing_field == :phi
			# density channel decouples the on-site (s-wave) interaction
			Us = Lattice_data.const_struct.U_params
		else
			error("Unknown pairing_field $(pairing_field)")
		end
		if Us[site] == 0.0
			return nothing, nothing, nothing, nothing
		end

		# phi is a real field; the pairing fields carry a complex step.
		field_update_type = pairing_field == :phi ? "real" : update_type
		#suggest update
		update_suggestion = suggest_update(Lattice_data; update_type=field_update_type)
		old_S = get_old_S(Lattice_data, method; site=site)
		update_Delta!(Lattice_data, site, pairing_field, pairing_symmetry, update_suggestion)
		new_S, new_energies, new_U_matrix = get_new_S(Lattice_data, method; site=site)
		dS = new_S - old_S

		return dS, update_suggestion, new_energies, new_U_matrix
	end

	"""
	    get_old_S(Lattice_data, method; site=1)

	Action before an update: the stored S (`"exact"`) or S_K0 (`"K0"`), or the
	travelling-cluster action around `site` (`"K0-TCA"`).
	"""
	function get_old_S(Lattice_data::LatticeData, method::String; site::Int64=1)
		if method == "exact"
			return Lattice_data.current_state.S
		elseif method == "K0"
			return Lattice_data.current_state.S_K0
		elseif method == "K0-TCA"
			return MCMF_action_TCluster_K0(Lattice_data, site)[1]
		elseif method == "TCluster"
			error("diag_method=\"TCluster\" (travelling cluster on the supercell lattice) is not " *
				  "implemented any more. Use diag_method=\"K0-TCA\" (with SC_L=1).")
		else
			error("unknown diag_method=\"$method\". Use \"K0\" or \"K0-TCA\".")
		end
	end

	"""
	    get_new_S(Lattice_data, method; site=1) -> (S, Energies, U_matrix)

	Action of the current (just changed) fields by `method`: `MCMF_action_K` (`"exact"`),
	`MCMF_action_K0` (`"K0"`) or `MCMF_action_TCluster_K0` around `site` (`"K0-TCA"`).
	"""
	function get_new_S(Lattice_data::LatticeData, method::String; site::Int64=1)
		if method == "exact"
			return MCMF_action_K(Lattice_data, Lattice_data.current_state.Delta_s, Lattice_data.current_state.Delta_dx, Lattice_data.current_state.Delta_dy)
		elseif method == "K0"
			return MCMF_action_K0(Lattice_data, Lattice_data.current_state.Delta_s, Lattice_data.current_state.Delta_dx, Lattice_data.current_state.Delta_dy)
		elseif method == "K0-TCA"
			return MCMF_action_TCluster_K0(Lattice_data, site)
		elseif method == "TCluster"
			error("diag_method=\"TCluster\" (travelling cluster on the supercell lattice) is not " *
				  "implemented any more. Use diag_method=\"K0-TCA\" (with SC_L=1).")
		else
			error("unknown diag_method=\"$method\". Use \"K0\" or \"K0-TCA\".")
		end
	end


	"""
	    update_Delta!(Lattice_data, site, pairing_field, pairing_symmetry, step)

	Add `step` to `pairing_field` (`:Delta_s`, `:Delta_dx`, `:Delta_dy` or `:phi`) at `site`,
	keeping the constraint of the pairing symmetry: `"mayr-d-wave"` keeps |Δ_dx| = |Δ_dy|,
	`"d-p-wave"` moves the other bond field by -step.
	"""
	function update_Delta!(Lattice_data::LatticeData, site::Int64, pairing_field::Symbol, pairing_symmetry::String, update_suggestion::Number)
		state =Lattice_data.current_state
		field = getfield(state, pairing_field)
		field[site] += update_suggestion
		field_value = field[site]

		if pairing_symmetry == "mayr-d-wave"
			# update Delta_D while keeping |Delta_dx| = |Delta_dy|
			# set the opposite field to have the same magnitude
			if pairing_field == :Delta_dx
				opposite_field = getfield(state, :Delta_dy)
				opposite_field[site] = abs(field_value) * cis(angle(opposite_field[site]))
			elseif pairing_field == :Delta_dy
				opposite_field = getfield(state, :Delta_dx)
				opposite_field[site] = abs(field_value) * cis(angle(opposite_field[site]))
			end
			
		elseif pairing_symmetry == "d-p-wave"
			# update Delta_dx and Delta_dy with opposite signs
			if pairing_field == :Delta_dx
				getfield(state, :Delta_dy)[site] += -update_suggestion
			elseif pairing_field == :Delta_dy
				getfield(state, :Delta_dx)[site] += -update_suggestion
			end
		end
		return nothing

	end

	"""
	    check_acceptence(dS) -> Bool

	Metropolis test: accept with probability min(1, e^{-ΔS}).
	"""
	function check_acceptence(dS::Float64)
		accepted = false
		if dS < 0
			#lowers S => accepted
			accepted = true
		else
			#does not lower S
			#accept or reject by the probability P=e^-dS
			P = exp(-dS)
			if rand() < P
				accepted = true
			end
		end
		return accepted
	end

	"""
	    revert_pairing_field!(Lattice_data, site, pairing_symmetry, pairing_field, step)

	Undo `update_Delta!` after a rejected update (same symmetry constraints).
	"""
	function revert_pairing_field!(Lattice_data::LatticeData, site::Int64, pairing_symmetry::String, pairing_field::Symbol, suggestion::Number)
		state =Lattice_data.current_state
		field = getfield(state, pairing_field)
		field[site] += -suggestion
		field_value = field[site]

		if pairing_symmetry == "mayr-d-wave"
			# revert Delta_D while keeping |Delta_dx| = |Delta_dy|
			# set the opposite field to have the same magnitude
			if pairing_field == :Delta_dx
				opposite_field = getfield(state, :Delta_dy)
				opposite_field[site] = abs(field_value) * cis(angle(opposite_field[site]))
			elseif pairing_field == :Delta_dy
				opposite_field = getfield(state, :Delta_dx)
				opposite_field[site] = abs(field_value) * cis(angle(opposite_field[site]))
			end
			
		elseif pairing_symmetry == "d-p-wave"
			# revert Delta_dx and Delta_dy with opposite signs
			if pairing_field == :Delta_dx
				getfield(state, :Delta_dy)[site] += suggestion
			elseif pairing_field == :Delta_dy
				getfield(state, :Delta_dx)[site] += suggestion
			end
		end
		return nothing

	end

	"""
	    Thermalize!(Lattice_data; method="exact", savedata="true", dir_path=nothing,
	                dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing)

	Run MC sweeps until `step_count` exceeds `num_thermal` (continues where a resumed run
	stopped). With `savedata`, saves progress every `saving_interval` steps and whenever
	`dump_time` has passed since the last save.
	"""
	function Thermalize!(Lattice_data::LatticeData; method="exact", savedata="true", dir_path=nothing, dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing)
		sim_data = Lattice_data.sim_data
        num_thermal = sim_data.num_thermal
		saving_interval = sim_data.saving_interval
		loop_random = sim_data.loop_random

		while Lattice_data.current_state.step_count <= num_thermal
		 	#MC step
			step_t0 = time()
            MC_stepV2!(Lattice_data; method=method, pairing=Lattice_data.sim_data.pairing_symmetry, loop_random=loop_random) #updating lattice
			step_elapsed = max(0.0, time() - step_t0)
			Lattice_data.current_state.timing_mc_step_total_sec += step_elapsed
			Lattice_data.current_state.timing_mc_step_count += 1

			if (mod(Lattice_data.current_state.step_count - num_thermal, saving_interval)==0 || (Dates.now() - Lattice_data.current_state.last_dump_time >= dump_time)) && savedata

				# save progress
				save_progress(Lattice_data, dir_path; sim_info_path=sim_info_path, log_filename=log_filename, saveplots=false, thermalized=false)
			end
		end
	end


	"""
	    MeasurmentSteps!(Lattice_data; method="exact", savedata=true, saving_interval=100, saveplots=false,
	                     dir_path=nothing, dump_time=Dates.Hour(1), sim_info_path=nothing,
	                     log_filename=nothing, n_only=false, fast_measurement=true)

	Run MC sweeps and measure every `measuring_interval` sweeps until `num_measure`
	measurements are done (continues a resumed run). With `savedata`, saves progress every
	`saving_interval` measurements, when the buffer is full, or when `dump_time` has passed,
	and writes a field snapshot every `snapshot_interval` measurements.
	`n_only=true` measures only n (μ search; cannot be combined with `savedata`);
	`fast_measurement` is passed to `make_measurement_K!` as `fast`.
	"""
	function MeasurmentSteps!(Lattice_data::LatticeData; method="exact", savedata=true, saving_interval=100, saveplots=false, dir_path=nothing, dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing, n_only::Bool=false, fast_measurement::Bool=true)
		n_only&& savedata && error("n_only measurements fill only n and Ss; they cannot be saved (savedata=true)")
		sim_data = Lattice_data.sim_data
		num_measure = sim_data.num_measure
		measuring_interval = sim_data.measuring_interval
		just_saved = false
		loop_random = sim_data.loop_random
		while Lattice_data.current_state.num_current_measurments+1 <= num_measure
			#MC step
			step_t0 = time()
			MC_stepV2!(Lattice_data; method = method, pairing=Lattice_data.sim_data.pairing_symmetry, loop_random=loop_random) #updating lattice
			step_elapsed = max(0.0, time() - step_t0)
			Lattice_data.current_state.timing_mc_step_total_sec += step_elapsed
			Lattice_data.current_state.timing_mc_step_count += 1

			if ((Lattice_data.current_state.step_count - sim_data.num_thermal) % measuring_interval) == 0
				#make measurment
				Lattice_data.current_state.num_current_measurments += 1
				Lattice_data.current_state.saving_measurement_ix += 1
				measurement_t0 = time()
				if n_only
					Lattice_data.measurement_data = make_measurement_n_only!(Lattice_data, Lattice_data.current_state.saving_measurement_ix)
				else
					Lattice_data.measurement_data = make_measurement_K!(Lattice_data, Lattice_data.current_state.saving_measurement_ix; fast=fast_measurement)
				end
				measurement_elapsed = max(0.0, time() - measurement_t0)
				Lattice_data.current_state.timing_measurement_total_sec += measurement_elapsed
				Lattice_data.current_state.timing_measurement_count += 1
				just_saved = false

				# Periodic raw-field snapshot -> separate field_snapshots.jld2.
				# Reads sim_data.snapshot_interval directly (0 => OFF). Captures the
				# just-measured MC configuration; no diag sync needed (HS fields are
				# always current).
				snap_int = sim_data.snapshot_interval
				if savedata && dir_path !== nothing && snap_int > 0 &&
				   (Lattice_data.current_state.num_current_measurments % snap_int == 0)
					save_field_snapshot(Lattice_data, dir_path)
				end
			end

			if ((mod(Lattice_data.current_state.num_current_measurments ,saving_interval)==0 && !just_saved) || (Dates.now() - Lattice_data.current_state.last_dump_time >= dump_time) || (Lattice_data.current_state.saving_measurement_ix == Lattice_data.sim_data.saving_interval)) && savedata 
				# save data
				if Lattice_data.current_state.saving_measurement_ix == 0
					continue
				end
				save_progress(Lattice_data, dir_path; sim_info_path=sim_info_path, log_filename=log_filename, saveplots=saveplots, thermalized=true)
				just_saved = true
			end
		end
	end

	"""
	    save_progress(Lattice_data, dir_path; sim_info_path=nothing, log_filename=nothing,
	                  saveplots=false, thermalized=true)

	Save a running simulation: bring the eigen-solution in sync, write checkpoint and
	measurements (`dump_measurements`), update this seed's tracking entry in sim.jld2
	(steps, measurements, timing, last save time), append a progress line to the log, and
	optionally save the quick-look plots.
	"""
	function save_progress(Lattice_data::LatticeData, dir_path::String; sim_info_path=nothing, log_filename=nothing, saveplots=false, thermalized=true)
		seed =Lattice_data.sim_data.seed
		# Ensure dumped checkpoints always contain a globally synchronized state.
		ensure_global_diag_synced!(Lattice_data; reason="save_progress")

		# save the lattice data
		savetime = Dates.now()
		Lattice_data.current_state.last_dump_time = savetime
		dump_measurements(Lattice_data, dir_path)
		
		# save the simulation info
		sim_info_path = sim_info_path === nothing ? joinpath(dir_path, "sim.jld2") : sim_info_path
		if isfile(sim_info_path)
			sim_info = JLD2.load(sim_info_path)
			# Ensure simulations dict exists
			if !haskey(sim_info, "simulations")
				sim_info["simulations"] = Dict{Any, Any}()
			end
			if !haskey(sim_info["simulations"], seed)
				sim_info["simulations"][seed] = Dict{String, Any}()
			end
			seed_dict = sim_info["simulations"][seed]
			seed_start_time = haskey(seed_dict, "start_time") ? tracking_datetime_or(savetime, seed_dict["start_time"]) : savetime
			ensure_seed_tracking_fields!(seed_dict, seed_start_time)

			segment_start = tracking_datetime_or(seed_start_time, seed_dict["timing_segment_start"])
			last_accounted = tracking_datetime_or(segment_start, seed_dict["timing_last_accounted_time"])
			base_time = last_accounted >= segment_start ? last_accounted : segment_start
			delta_active_sec = elapsed_seconds(base_time, savetime)
			seed_dict["timing_active_sec"] = tracking_float_or(0.0, get(seed_dict, "timing_active_sec", 0.0)) + delta_active_sec
			seed_dict["timing_last_accounted_time"] = savetime

			seed_dict["step_count"] = Lattice_data.current_state.step_count
			seed_dict["num_current_measurments"] = Lattice_data.current_state.num_current_measurments
			seed_dict["timing_mc_step_total_sec"] = Lattice_data.current_state.timing_mc_step_total_sec
			seed_dict["timing_mc_step_count"] = Lattice_data.current_state.timing_mc_step_count
			seed_dict["timing_measurement_total_sec"] = Lattice_data.current_state.timing_measurement_total_sec
			seed_dict["timing_measurement_count"] = Lattice_data.current_state.timing_measurement_count
			seed_dict["last_save_time"] = savetime
			sim_info["simulations"][seed] = seed_dict
			JLD2.save(sim_info_path, sim_info)
		end

		# update the log file
		log_filename = log_filename === nothing ? joinpath(dir_path, "simulation_log.txt") : log_filename
		total_steps = Lattice_data.current_state.step_count
		if !thermalized
			num_thermal = Lattice_data.sim_data.num_thermal
			open(log_filename, "a") do io
				println(io, "-saving progress- $(savetime)")
				println(io, "-> Steps Taken : $(total_steps)/$(num_thermal)=$(total_steps/num_thermal * 100)% Thermalized")
			end
		else
			c_mes = Lattice_data.current_state.num_current_measurments
			num_measure = Lattice_data.sim_data.num_measure
			open(log_filename, "a") do io
				println(io, "-saving progress- $(savetime)")
				println(io, "-> Measurments taken : $(c_mes)/$(num_measure)=$(c_mes/num_measure * 100)%")
				println(io, "-> Total steps taken : $(total_steps)")
			end
		end

		if saveplots 
			save_all_plots(Lattice_data; directory=dir_path)
		end
	end


    """
        MonteCarloSimulation(sim_data::SimData, dir_path; kwargs...) -> LatticeData
        MonteCarloSimulation(Lattice_data::LatticeData, dir_path; saveplots=false, savedata=true,
                             saving_interval=100, dump_time=Dates.Hour(1), sim_info_path=nothing,
                             log_filename=nothing, n_only=false, fast_measurement=true) -> LatticeData

    Run (or continue) one simulation: check memory and allocate the measurement buffers
    (not for `n_only`), thermalize (`Thermalize!`), then measure (`MeasurmentSteps!`),
    saving into `dir_path` if `savedata`. The `SimData` method starts from a new
    `Lattice_Data(sim_data)`. A resumed run that is already measuring skips the
    end-of-thermalization save.
    """
    function MonteCarloSimulation(sim_data::SimData, dir_path::String; saveplots=false, savedata=true, saving_interval=100, dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing, n_only::Bool=false, fast_measurement::Bool=true)
        #make initial fields
        Lattice_data = Lattice_Data(sim_data) # make initial lattice
		Lattice_data = MonteCarloSimulation(Lattice_data, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename, n_only=n_only, fast_measurement=fast_measurement) # update the lattice with initial fields

        return Lattice_data
    end


	function MonteCarloSimulation(Lattice_data::LatticeData, dir_path::String; saveplots=false, savedata=true, saving_interval=100, dump_time=Dates.Hour(1), sim_info_path=nothing, log_filename=nothing, n_only::Bool=false, fast_measurement::Bool=true)
		# n_only=true: the occupation-convergence phase; only <n> is measured and the
		# measurement buffers are never allocated (see make_measurement_n_only!).
		n_only && savedata && error("n_only measurements fill only n and Ss; they cannot be saved (savedata=true)")
		sim_data = Lattice_data.sim_data
		if !n_only
			# Fail now, not after thermalization, if the run cannot fit in memory
			# (override: MC_SKIP_MEMORY_CHECK=1), then allocate the measurement buffers.
			check_memory_budget(sim_data; fast=fast_measurement, verbose=savedata)
			ensure_measurement_buffers!(Lattice_data; fast=fast_measurement)
		end
		# Make sure the field-histogram accumulators match the configured bin counts
		# (initializes them for legacy checkpoints without touching a live run's counts).
		ensure_field_histograms!(Lattice_data)
		started_with_measurements = Lattice_data.current_state.num_current_measurments > 0

		#thermalize
		Thermalize!(Lattice_data; method=sim_data.diag_method, savedata=savedata, dir_path=dir_path, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
		if savedata && !started_with_measurements
			save_progress(Lattice_data, dir_path; sim_info_path=sim_info_path, log_filename=log_filename, saveplots=false, thermalized=false)
			# save log
			log_filename = log_filename === nothing ? joinpath(dir_path, "simulation_log.txt") : log_filename
			open(log_filename, "a") do io
				println(io, "-Thermalization ended!- $(Time(Dates.now()))")
			end
		elseif savedata && started_with_measurements
			log_filename = log_filename === nothing ? joinpath(dir_path, "simulation_log.txt") : log_filename
			open(log_filename, "a") do io
				println(io, "-Resume detected: skipping thermalization-end save (already in measurement phase)- $(Time(Dates.now()))")
			end
		end

        #measure
		MeasurmentSteps!(Lattice_data; method=sim_data.diag_method, savedata=savedata, saving_interval=saving_interval, saveplots=saveplots, dir_path=dir_path, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename, n_only=n_only, fast_measurement=fast_measurement)
		if savedata
			save_progress(Lattice_data, dir_path; sim_info_path=sim_info_path, log_filename=log_filename, saveplots=saveplots, thermalized=true)
		end
		return Lattice_data
	end

end

"""
    get_num_current_measurment(lattice_data) -> Int

Number of measurements taken so far.
"""
function get_num_current_measurment(lattice_data::LatticeData)

	return lattice_data.current_state.num_current_measurments
end

begin ## plotting functions
	# Quick-look plots of ONE run in memory (a LatticeData), for interactive checks.
	# Thesis figures are made from means_db, not from these.

	"""
	    plot_DOS(lattice_data; avrage=true)

	Plot the DOS(ω): averaged over the measurements, or every measurement (`avrage=false`).
	"""
	function plot_DOS(lattice_data::LatticeData; avrage=true)
        sim_data = lattice_data.sim_data
		num_current_measurment = get_num_current_measurment(lattice_data)
		if avrage
			DOS = sum(lattice_data.measurement_data.DOS[1:num_current_measurment])/num_current_measurment
			tit = "<DOS>"
		else 
			DOS = lattice_data.measurement_data.DOS[1:num_current_measurment]
			tit = "all DOS"
		end

		fig = Plots.plot(lattice_data.sim_data.w_list, DOS, legend=false,xlabel="w",ylabel="n(w)", title=tit)
	end

	
	"""
	    plot_mcmf_S(lattice_data)

	Plot the action S at each measurement.
	"""
	function plot_mcmf_S(lattice_data::LatticeData)
		num_current_measurment = get_num_current_measurment(lattice_data)
		
		Plots.plot(lattice_data.measurement_data.Ss[1:num_current_measurment], title="S_MCMF", legend=false)
	end


	
    """
        plot_spectral_function(lattice_data)

    Heatmap of the mean A(k, ω) along the k path (0,0) → (π,0) → (π,π) → (0,0).
    """
    function plot_spectral_function(lattice_data::LatticeData)
        sim_data = lattice_data.sim_data
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		L_S = sim_data.SC_L
		Lx_gen = Lx * L_S
		Ly_gen = Ly * L_S
		L_gen_max = max(Lx_gen, Ly_gen)
		w_list = lattice_data.sim_data.w_list

		# Spectral function k-path length: seg1 + seg2 + seg3 + 1 (final (0,0) point)
		path_len = 2 * (Lx_gen - div(Lx_gen, 2)) + (Ly_gen - div(Ly_gen, 2)) + 1

		# A is Array{Array{Float64,1},1} with A[w_idx][k_path_idx]
		# hcat creates matrix of shape (path_len, num_w)
		# Transpose to get (num_w, path_len) for heatmap with k on x-axis and w on y-axis
		A = Matrix(hcat(lattice_data.measurement_data.A...)')
    
		# Create k-path labels for rectangular lattice
		# Always use index 1 for π (both even and odd L)
		# Path: (0,0) -> (π,0) -> (π,π) -> (0,0)
		seg1_len = Lx_gen - div(Lx_gen, 2)  # (0,0) to (π,0)
		seg2_len = Ly_gen - div(Ly_gen, 2)  # (π,0) to (π,π)
		seg3_len = Lx_gen - div(Lx_gen, 2)  # (π,π) to (0,0)
		
		# Triangular path: (0,0) → (π,0) → (π,π) → (0,0)
		k_names = fill("", path_len)
		k_names[1] = "(0,0)"
		k_names[seg1_len + 1] = "(π,0)"
		k_names[seg1_len + seg2_len + 1] = "(π,π)"
		k_names[path_len] = "(0,0)"
		
		# Plot heatmap
        Plots.heatmap(1:path_len, w_list, reverse(A, dims=2), xlabel="(kx,ky)", ylabel="w", title="<A(w,k)>, beta=$(sim_data.beta), $(Lx)x$(Ly)", xticks=(1:path_len, k_names))
    end

	"""
	    plot_BZ(data)

	Heatmap of the mean A(k, ω = 0) over the Brillouin zone (the Fermi surface).
	"""
	function plot_BZ(data::LatticeData)
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		L_sc = data.sim_data.SC_L
		Lx_gen = Lx * L_sc
		Ly_gen = Ly * L_sc
		BZ =data.measurement_data.BZ
		# Reshape to (Ly_gen, Lx_gen) - ky varies along rows, kx along columns
		# This matches heatmap convention: first dim = y-axis, second dim = x-axis
		BZ_matrix = reshape(BZ, Ly_gen, Lx_gen)
		# Create proper axis labels (k values from 0 to 2π)
		kx_vals = (0:Lx_gen-1) .* (2π / Lx_gen)
		ky_vals = (0:Ly_gen-1) .* (2π / Ly_gen)
		Plots.heatmap(kx_vals, ky_vals, BZ_matrix, xlabel="kx", ylabel="ky", 
			title="Brillouin Zone $(Lx)x$(Ly), n=$(round(mean(data.measurement_data.n), digits=2)), beta=$(data.sim_data.beta)")
	end
	

	
	"""
	    plot_n(data)

	Plot the filling n at each measurement.
	"""
	function plot_n(data::LatticeData)
		num_current_measurment = get_num_current_measurment(data)
		
		n = data.measurement_data.n[1:num_current_measurment]
		Plots.plot(n, legend=false, xlabel="MC step", ylabel="<n>", title="<n>, $(data.sim_data.Lx)x$(data.sim_data.Ly), mu=$(data.sim_data.mu), beta=$(data.sim_data.beta)")
	end

	"""
	    plot_Kx(data)

	Plot the x kinetic energy Kx at each measurement.
	"""
	function plot_Kx(data::LatticeData)
		num_current_measurment = get_num_current_measurment(data)
		
		Kx = data.measurement_data.Kx[1:num_current_measurment]
		Plots.plot(Kx, legend=false, xlabel="MC step", ylabel="<Kx>", title="-0.25*<Kx>, mu =$(data.sim_data.mu), beta=$(data.sim_data.beta)")
	end

	"""
	    plot_Lxx(data; qi="y", new=false)

	Plot Λxx(q, ω = 0) at each measurement, one line per q ≠ 0, for q along y (`qi="y"`)
	or x (`qi="x"`). `new=true` reads `Lxx_q*_new`, which no longer exists (errors).
	"""
	function plot_Lxx(data::LatticeData; qi="y", new=false)
		num_current_measurment = get_num_current_measurment(data)
		
		if qi == "y"
			if new
				Lxx = data.measurement_data.Lxx_qy_new[1:num_current_measurment]
			else
				Lxx = data.measurement_data.Lxx_qy[1:num_current_measurment]
			end
			label = "Λ_xx(qx=0, qy, w=0)"
		elseif qi == "x"
			if new
				Lxx = data.measurement_data.Lxx_qx_new[1:num_current_measurment]
			else
				Lxx = data.measurement_data.Lxx_qx[1:num_current_measurment]
			end
		
			label = "Λ_xx(qx, qy=0, w=0)"
		end
		
		Lxx_mat = hcat(Lxx...)
		fig = Plots.plot(xlabel="MC step", ylabel=label, title="Λ_xx(q$(qi), ω=0), mu =$(data.sim_data.mu), beta=$(data.sim_data.beta)")
		for i in eachindex(Lxx[1])
			if i ==1
				continue # skip the first element, which is q=0
			end
			
			Plots.plot!(fig, Lxx_mat[i,:], label="q$(qi)*N$(qi)/2π=$(i-1)")
		end
		return fig
	end


	"""
	    plot_Lxx_means(data; y_lims=nothing, std_scheme="correlation", direction1="x", direction2="x")

	Plot the mean Λab(q, ω = 0) ± error against q, for q along y and along x, with -<K>
	at q = 0 for the diagonal components (the two limits that give the stiffness and
	the longitudinal sum rule). `direction1 = "both"` (xx and yy) or `"all"` (four
	components) makes a panel figure.
	"""
	function plot_Lxx_means(data::LatticeData; y_lims=nothing, std_scheme="correlation", direction1="x", direction2="x")
		num_current_measurment = get_num_current_measurment(data)
		if (direction1,direction2) == ("x","x")
			Lxx_qy = [data.measurement_data.Lxx_qy[i][1:end] for i in 1:num_current_measurment]
			Lxx_qx = [data.measurement_data.Lxx_qx[i][1:end] for i in 1:num_current_measurment]
			Kx = data.measurement_data.Kx[1:num_current_measurment]
			L_name = "Λxx"
			K_name = "Kx"
		elseif (direction1,direction2) == ("x","y")
			Lxx_qy = [data.measurement_data.Lxy_qy[i][1:end] for i in 1:num_current_measurment]
			Lxx_qx = [data.measurement_data.Lxy_qx[i][1:end] for i in 1:num_current_measurment]
			L_name = "Λxy"
				K_name = "Kx"
		elseif (direction1,direction2) == ("y","y")
			Lxx_qy = [data.measurement_data.Lyy_qy[i][1:end] for i in 1:num_current_measurment]
			Lxx_qx = [data.measurement_data.Lyy_qx[i][1:end] for i in 1:num_current_measurment]
			Kx = data.measurement_data.Ky[1:num_current_measurment]
			L_name = "Λyy"
			K_name = "Ky"
		elseif (direction1,direction2) == ("y","x")
			Lxx_qy = [data.measurement_data.Lyx_qy[i][1:end] for i in 1:num_current_measurment]
			Lxx_qx = [data.measurement_data.Lyx_qx[i][1:end] for i in 1:num_current_measurment]
			Kx = data.measurement_data.Ky[1:num_current_measurment]
			L_name = "Λyx"
			K_name = "Ky"
		elseif direction1=="both"
			fig1 = plot_Lxx_means(data; std_scheme=std_scheme, direction1="x", direction2="x")
			fig2 = plot_Lxx_means(data; std_scheme=std_scheme, direction1="y", direction2="y")
			fig = plot(fig1, fig2, layout=(2,1))
			return fig
		elseif direction1=="all"
			fig1 = plot_Lxx_means(data; std_scheme=std_scheme, direction1="x", direction2="x")
			fig2 = plot_Lxx_means(data; std_scheme=std_scheme, direction1="y", direction2="y")
			fig3 = plot_Lxx_means(data; std_scheme=std_scheme, direction1="x", direction2="y")
			fig4 = plot_Lxx_means(data; std_scheme=std_scheme, direction1="y", direction2="x")
			fig = plot(fig1, fig2, fig3, fig4, layout=(2,2))
			return fig
		else
			error("Invalid direction combination: $(direction1), $(direction2). Use either (x,x), (x,y), (y,x) or (y,y).")
		end


		Lxx_qy_mean, Lxx_qy_err = observable_mean(Lxx_qy,num_current_measurment; std_scheme=std_scheme)
		Lxx_qx_mean, Lxx_qx_err = observable_mean(Lxx_qx,num_current_measurment; std_scheme=std_scheme)
		
		n_avrage = mean(data.measurement_data.n[1:num_current_measurment])

		fig = Plots.plot(xlabel="qi*Ni/2π", ylabel=L_name, title="$(L_name), <n>=$(round(n_avrage, digits=2)) , beta=$(data.sim_data.beta)")
		Plots.plot!(fig, Lxx_qy_mean, yerr=Lxx_qy_err, label="$(L_name)(qx=0, qy, w=0)", linewidth=2, marker=:circle, color=:blue)
		Plots.plot!(fig, Lxx_qx_mean, yerr=Lxx_qx_err, label="$(L_name)(qx, qy=0, w=0)", linewidth=2, marker=:square, color=:green)
		if direction1 == direction2
			Kx_mean, Kx_std = observable_mean(Kx, num_current_measurment; std_scheme=std_scheme)
			Plots.scatter!(fig, [0], [-Kx_mean], yerr=[Kx_std], label="<-$K_name>", marker=:star, color=:red, markersize=8)
		end
		if y_lims !== nothing
			Plots.ylim!(fig, y_lims)
		end
		return fig
	end


	"""
	    plot_conductivity_regular_w(data)

	Old plot of σ(ω). Broken: reads `measurement_data.conductivity_regular`, which no
	longer exists.
	"""
	function plot_conductivity_regular_w(data::LatticeData)
		num_current_measurment = get_num_current_measurment(data)
		cond_reg = data.measurement_data.conductivity_regular[1:num_current_measurment]
		cond_val, cond_err = observable_mean(cond_reg, num_current_measurment; std_scheme="correlation")		

		w_list = data.sim_data.w_conductivity_range
		beta = data.sim_data.beta
		n = mean(data.measurement_data.n[1:num_current_measurment])

		fig = Plots.plot(xlabel="w", ylabel="σ(w)", title="Conductivity, beta=$(beta), n=$(round(n, sigdigits=2))")
		plot!(fig, w_list, cond_val, label="<σ(w)>", linewidth=2)
		plot!(fig, w_list, cond_val .+ cond_err, label=nothing, linewidth=1, linestyle=:dash, color=:blue)
		plot!(fig, w_list, cond_val .- cond_err, label=nothing, linewidth=1, linestyle=:dash, color=:blue)
		plot!(fig, ylims=(0.0, maximum(cond_val .+ cond_err)))
		return fig
	end


	#### plots with errors


	"""
	    plot_DOS_with_errors(data)

	Plot the mean DOS(ω) with its error band.
	"""
	function plot_DOS_with_errors(data::LatticeData)
		measurement_data = data.measurement_data

		num_current_measurment = get_num_current_measurment(data)

		w_list = data.sim_data.w_list
		beta = data.sim_data.beta
		DoS = measurement_data.DOS[1:num_current_measurment]

		Dos_avrage, DoS_error = observable_mean(DoS, num_current_measurment; std_scheme="correlation")
		fig = Plots.plot(w_list, Dos_avrage, color="blue", title="<DOS>, beta=$(beta)", xlabel="w", ylabel="n(w)", label="DOS", linewidth=3)
		Plots.plot!(w_list, Dos_avrage .+ DoS_error, color="red", title="<DOS>, beta=$(beta)", xlabel="w", ylabel="n(w)", label=nothing, linewidth=1, linestyle=:dash)
		Plots.plot!(w_list, Dos_avrage .- DoS_error, color="red", title="<DOS>, beta=$(beta)", xlabel="w", ylabel="n(w)", label=nothing, linewidth=1, linestyle=:dash)

		fig
	end


    """
        plot_stiffness(data)

    Plot -¼(Kx + Λxx(qx=0, smallest qy)) at each measurement (a rough per-measurement
    stiffness; the reported one is extrapolated to q = 0 in `get_means`).
    """
    function plot_stiffness(data::LatticeData)
        num_measure = data.sim_data.num_measure
		num_current_measurment = get_num_current_measurment(data)
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		stiffness = [-0.25*(data.measurement_data.Kx[i] + data.measurement_data.Lxx_qy[i][1]) for i in 1:num_current_measurment]
		fig = Plots.plot(stiffness, title="stiffness, $(Lx)x$(Ly), beta=$(data.sim_data.beta)", xlabel="MC step", ylabel="stiffness")
		
        return fig

    end

	"""
	    plot_conductivity(data)

	Old plot of σ at each measurement. Broken: reads `measurement_data.conductivity`,
	which no longer exists.
	"""
	function plot_conductivity(data::LatticeData)
		num_current_measurment = get_num_current_measurment(data)
		fig = Plots.plot(data.measurement_data.conductivity[1:num_current_measurment], title="conductivity, $(data.sim_data.Lx)x$(data.sim_data.Ly), beta=$(data.sim_data.beta)", xlabel="MC step", ylabel="sigma")
		return fig
	end


	
	"""
	    plot_Delta(Delta; Lx=0, Ly=0)

	Heatmap of |Δ_i| with an arrow per site showing the phase of Δ_i. Without `Lx`, `Ly`
	the lattice is taken as square. Switches Plots to the GR backend.
	"""
	function plot_Delta(Delta::Array{ComplexF64,1}; Lx::Int64=0, Ly::Int64=0)
		# If Lx, Ly not provided, assume square lattice
		N =length(Delta)
		if Lx == 0 || Ly == 0
			Lx = Int64(sqrt(N))
			Ly = Lx
		end
		# Reshape Delta to a 2D grid and transpose so x is on x-axis
		# Original indexing: Delta[idx(x,y,Lx)] where idx = (y-1)*Lx + x
		# reshape gives (Lx, Ly) with first dim = x, second dim = y
		# Transpose so heatmap shows x on x-axis, y on y-axis
		Delta_2D = reshape(Delta, Lx, Ly)
		Delta_abs = abs.(Delta_2D)'
		Delta_phase = angle.(Delta_2D)

		# Create grid for arrows - need (x, y) coordinates matching heatmap pixels
		# When heatmap is called with explicit x=1:Lx and y=1:Ly, pixels are centered at integer values
		x_coords = Float64[]
		y_coords = Float64[]
		u = Float64[]
		v = Float64[]
		for ix in 1:Lx
			for iy in 1:Ly
				push!(x_coords, Float64(ix))  # Center of pixel at integer coordinate
				push!(y_coords, Float64(iy))  # Center of pixel at integer coordinate
				phase = Delta_phase[ix, iy]
				push!(u, 0.4 * cos(phase))
				push!(v, 0.4 * sin(phase))
			end
		end
		
		# Plot heatmap of |Delta|
		gr()  # Switch to GR backend for proper quiver arrow support
		fig = Plots.heatmap(1:Lx, 1:Ly, Delta_abs, xlabel="x", ylabel="y", 
			title="|Delta| Heatmap with Phase Arrows", color=:viridis)
		Plots.quiver!(x_coords, y_coords, quiver=(u, v), color=:black, lw=1.5, label="", arrow=(:head, 0.5, 0.3)) # add quiver arrow heads

			
		return fig
	end


	
	"""
	    plot_mean_delta(data)

	Plot |mean of the Δ field| and |⟨Δ̂⟩| (the pair operator) at each measurement, with the
	clean mean-field gap as a dashed line.
	"""
	function plot_mean_delta(data::LatticeData)
		num_measure = get_num_current_measurment(data)
		delta_mean = abs.(data.measurement_data.Delta_mean[1:num_measure])
		Delta_operator = abs.(data.measurement_data.Delta_operator[1:num_measure])
		mean_n = mean(data.measurement_data.n[1:num_measure])
		my_label = "|<Δ>|"
		my_tit = "|<Δ>|, beta=$(data.sim_data.beta), <n>=$(round(mean_n, digits=2))"

		fig =  Plots.plot(delta_mean, label=my_label, color=:blue)
		Plots.plot!(fig,[0,num_measure],abs(saddle_point_delta(data.sim_data)[1])*[1,1], label="saddle point delta", color=:red, linestyle=:dash)
		Plots.plot!(fig, Delta_operator, label="|<Δ̂>| = |∑<cc>|")
		Plots.plot!(fig, title=my_tit, xlabel="measurements", ylabel=my_label, legend=:topright, grid=true)
		
		return fig
	end

	"""
	    plot_mean_delta_phase(data)

	Plot the phase of the mean Δ field at each measurement.
	"""
	function plot_mean_delta_phase(data::LatticeData)
		num_current_measurment = get_num_current_measurment(data)
		mean_phase = angle.(data.measurement_data.Delta_mean[1:num_current_measurment])
		fig =  Plots.plot(1:num_current_measurment, mean_phase, label="Delta")
		Plots.plot!(title="phase(Delta)", xlabel="measurements", ylabel="angle(Delta)", legend=:topright, grid=true)
		
		return fig
	end

	"""
	    plot_Ps(data)

	Plot Ps = 2 DdD - n + 1 (pairing structure factor) at each measurement.
	"""
	function plot_Ps(data::LatticeData)
		num_current_measurment = get_num_current_measurment(data)
		Ps = [2 * data.measurement_data.DdD[i] - data.measurement_data.n[i] + 1 for i in 1:num_current_measurment]
		avr_n = mean(data.measurement_data.n[1:num_current_measurment])
		fig = Plots.plot(Ps, title="Ps, $(data.sim_data.Lx)x$(data.sim_data.Ly), <n>=$(round(avr_n, digits=2)), beta=$(data.sim_data.beta)", xlabel="MC step", ylabel="Ps", label=nothing) 

		return fig
	end


	"""
	    new_fig(; show=true)

	Empty figure with wide margins.
	"""
	function new_fig(;show=true)
    fig =Plots.plot(left_margin=12mm, right_margin=12mm, top_margin=10mm, bottom_margin=10mm, show=show)
    return fig
	end

	"""
	    set_fig!(fig)

	Give `fig` wide margins (for saving).
	"""
	function set_fig!(fig)
		Plots.plot!(fig, left_margin=12mm, right_margin=12mm, top_margin=10mm, bottom_margin=10mm)
		return fig
	end

	"""
	    save_all_plots(data; directory)

	Old version: stops with an error. It was meant to save the quick-look plots of one run
	as PNGs in `directory` (called by `save_progress` when `saveplots=true`), but it calls
	`DOS_binning_example`, which is not defined, and `plot_conductivity`, which reads a
	removed field. Keep `saveplots=false` (the default) until it is updated.
	"""
	function save_all_plots(data::LatticeData; directory=joinpath(pwd(), "plots"))
		error("save_all_plots is an old version that needs to be updated (it calls DOS_binning_example, " *
			  "which is not defined, and plot_conductivity, which reads a removed field). " *
			  "Run with saveplots=false, or call the plot_* functions directly.")
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		mu = data.sim_data.mu
		beta = data.sim_data.beta
		method = data.sim_data.diag_method

		sim_name = "$(Lx)x$(Ly)_beta=$(round(beta,digits=2))_mu=$(round(mu,digits=2))_meth=$(method).png"


		fig1 = plot_DOS_with_errors(data)
		Plots.savefig(set_fig!(fig1), joinpath(directory, "DOS_$(sim_name)"))
		fig2 = plot_mcmf_S(data)
		Plots.savefig(set_fig!(fig2), joinpath(directory, "S_MCMF_$(sim_name)"))
		fig3 = plot_spectral_function(data)
		Plots.savefig(set_fig!(fig3), joinpath(directory, "A_$(sim_name)"))
		fig4 = plot_BZ(data)
		Plots.savefig(set_fig!(fig4), joinpath(directory, "BZ_$(sim_name)"))
		fig5 = plot_n(data)	
		Plots.savefig(set_fig!(fig5), joinpath(directory, "n_$(sim_name)"))
		fig6 = DOS_binning_example(data)
		Plots.savefig(set_fig!(fig6), joinpath(directory, "DOS_binning_$(sim_name)"))
		fig7 = plot_conductivity(data)
		Plots.savefig(set_fig!(fig7), joinpath(directory, "conductivity_$(sim_name)"))
		fig8 = plot_mean_delta(data)
		Plots.savefig(set_fig!(fig8), joinpath(directory, "mean_delta_$(sim_name)"))
		fig9 = plot_mean_delta_phase(data)
		Plots.savefig(set_fig!(fig9), joinpath(directory, "mean_delta_phase_$(sim_name)"))
		fig10 = plot_Kx(data)
		Plots.savefig(set_fig!(fig10), joinpath(directory, "Kx_$(sim_name)"))
		fig11 = plot_Lxx(data; qi="y", new=false)
		Plots.savefig(set_fig!(fig11), joinpath(directory, "Lxx(qy)_$(sim_name)"))
		fig12 = plot_Lxx(data; qi="x", new=false)
		Plots.savefig(set_fig!(fig12), joinpath(directory, "Lxx(qx)_$(sim_name)"))
		fig13 = plot_Lxx_means(data)
		Plots.savefig(set_fig!(fig13), joinpath(directory, "Lxx_means_$(sim_name)"))
		fig14 = plot_Ps(data)
		Plots.savefig(set_fig!(fig14), joinpath(directory, "Ps_$(sim_name)"))
		fig15 = plot_stiffness(data)
        Plots.savefig(set_fig!(fig15), joinpath(directory, "stiffness_$(sim_name)"))
	end

end


begin # uniform guess example
	

	"""
	    saddle_point_plot(sim_data; d_min=0.0001)

	Old plot of the gap equation and the action against a uniform Δ. Broken: uses
	`delta_range` and `MCMF_action`, which are not defined here.
	"""
	function saddle_point_plot(sim_data::SimData; d_min=0.0001)

		delta_guess, eqs, min_ix = saddle_point_delta(sim_data; d_min=0.0001)
		N = sim_data.N

		Plots.plot(delta_range,eqs, label="eq.", xlabel="Delta", title="uniform Delta initial guess")
		Plots.scatter!([delta_guess], [eqs[min_ix]], label="solution")
		Plots.plot!(delta_range, [MCMF_action(sim_data, ones(N)*delta*(1+0im))[1] for delta in delta_range], label="MCMF action")
		
	end

end

begin # file name
    """
        choose_file_name(location, sim_data; date_folder=true, numbered=true, seed_folder=false) -> String

    Old single-file output name `DATA_<Lx>x<Ly>_beta=..._mu=..._method=...[_#k].jld2` under
    `location` (optionally in a date folder and a `seed_<s>` folder, created if needed;
    `numbered` picks the first unused #k). Not used by the current pipeline.
    """
    function choose_file_name(location::String, sim_data::SimData; date_folder=true, numbered=true, seed_folder=false)

        dir_path = location
		if date_folder
			# Create directory with current date
			date_str = Dates.format(Dates.now(), "dd-mm-yyyy")
        	dir_path = joinpath(location, date_str)
		end

		if seed_folder && sim_data.seed != 0 && sim_data.seed !== nothing
			# Create directory with seed
			dir_path = joinpath(dir_path, "seed_$(sim_data.seed)")
		end

        if !isdir(dir_path)
            mkpath(dir_path)
        end

        # Generate unique filename
        base_filename = "DATA_$(sim_data.Lx)x$(sim_data.Ly)_beta=$(round(sim_data.beta,digits=2))_mu=$(round(sim_data.mu,digits=2))_method=$(sim_data.diag_method)"
		filename = joinpath(dir_path, "$(base_filename).jld2")

		if numbered
			filename = joinpath(dir_path, "$(base_filename)_#1.jld2")
        	counter = 2
			while isfile(filename)
				filename = joinpath(dir_path, "$(base_filename)_#$(counter).jld2")
				counter += 1
			end
		end
        
        return filename
    end


	"""
	    choose_path(location, sim_data; date_folder=true, numbered=true, seed_folder=false) -> String

	Output folder of one run, created if needed: `location`, then a `dd-mm-yyyy` folder
	(`date_folder`), a `seed_<seed>` folder (`seed_folder`), and a new `#k` folder (`numbered`,
	first unused k).
	"""
	function choose_path(location::String, sim_data::SimData; date_folder=true, numbered=true, seed_folder=false)

        dir_path = location
		if date_folder
			# Create directory with current date
			date_str = Dates.format(Dates.now(), "dd-mm-yyyy")
        	dir_path = joinpath(location, date_str)
		end

		if seed_folder && sim_data.seed != 0 && sim_data.seed !== nothing
			# Create directory with seed
			dir_path = joinpath(dir_path, "seed_$(sim_data.seed)")
		end

        if !isdir(dir_path)
            mkpath(dir_path)
        end


		if numbered
			num_dir = joinpath(dir_path, "#1")
        	counter = 2
			while isdir(num_dir)
				num_dir = joinpath(dir_path, "#$(counter)")
				counter += 1
			end
			mkpath(num_dir)
			dir_path = num_dir
		end

        return dir_path
    end
end




begin # running the simulation - helper functions

	"""
	    setup_simulation_paths(directory, sim_data; date_folder=true, numbered=true, seed_folder=false)
	        -> (dir_path, sim_info_path, log_filename, checkpoint_path)

	Create the run folder (`choose_path`) and return the paths of its files. sim.jld2 is
	in `directory` (the parameter point), the log and checkpoint in the run folder.
	"""
	function setup_simulation_paths(directory::String, sim_data::SimData; 
									 date_folder::Bool=true, numbered::Bool=true, seed_folder::Bool=false)
		dir_path = choose_path(directory, sim_data; date_folder=date_folder, numbered=numbered, seed_folder=seed_folder)
		sim_info_path = joinpath(directory, "sim.jld2")
		log_filename = joinpath(dir_path, "simulation_log.txt")
		checkpoint_path = joinpath(dir_path, "checkpoint.jld2")
		
		return dir_path, sim_info_path, log_filename, checkpoint_path
	end

	# Per-seed tracking in sim.jld2 ("simulations" => seed => Dict): status, paths, start
	# and end times, and timing (active wall time, resumes, time in MC steps and
	# measurements). Old entries may store times as strings; the *_or helpers read them.

	"""
	    tracking_datetime_or(default, value) -> DateTime

	`value` as a DateTime (parsed if a string), else `default`.
	"""
	function tracking_datetime_or(default_dt::DateTime, value)
		if value isa DateTime
			return value
		elseif value isa AbstractString
			try
				return DateTime(value)
			catch
				return default_dt
			end
		end
		return default_dt
	end

	"""
	    tracking_float_or(default, value) -> Float64

	`value` as a Float64 (parsed if a string), else `default`.
	"""
	function tracking_float_or(default_val::Float64, value)
		if value isa Number
			return Float64(value)
		end
		parsed = tryparse(Float64, string(value))
		return parsed === nothing ? default_val : parsed
	end

	"""
	    tracking_int_or(default, value) -> Int

	`value` as an Int (rounded or parsed), else `default`.
	"""
	function tracking_int_or(default_val::Int, value)
		if value isa Integer
			return Int(value)
		end
		if value isa Number
			return Int(round(value))
		end
		parsed = tryparse(Int, string(value))
		return parsed === nothing ? default_val : parsed
	end

	"""
	    elapsed_seconds(start, stop) -> Float64

	Seconds from `start` to `stop` (0 if negative).
	"""
	function elapsed_seconds(start_dt::DateTime, end_dt::DateTime)
		return max(0.0, Dates.value(end_dt - start_dt) / 1000.0)
	end

	"""
	    ensure_seed_tracking_fields!(seed_dict, start_time)

	Add any missing timing fields to a seed's tracking entry (zeros, or times derived from
	`start_time` / `last_save_time`), so old entries can be updated like new ones.
	"""
	function ensure_seed_tracking_fields!(seed_dict::AbstractDict, start_time::DateTime)
		if !haskey(seed_dict, "timing_source")
			seed_dict["timing_source"] = "tracked_live"
		end
		if !haskey(seed_dict, "timing_active_sec")
			seed_dict["timing_active_sec"] = 0.0
		end
		if !haskey(seed_dict, "timing_resume_count")
			seed_dict["timing_resume_count"] = 0
		end
		if !haskey(seed_dict, "timing_mc_step_total_sec")
			seed_dict["timing_mc_step_total_sec"] = 0.0
		end
		if !haskey(seed_dict, "timing_mc_step_count")
			seed_dict["timing_mc_step_count"] = 0
		end
		if !haskey(seed_dict, "timing_measurement_total_sec")
			seed_dict["timing_measurement_total_sec"] = 0.0
		end
		if !haskey(seed_dict, "timing_measurement_count")
			seed_dict["timing_measurement_count"] = 0
		end

		segment_start = haskey(seed_dict, "timing_segment_start") ?
			tracking_datetime_or(start_time, seed_dict["timing_segment_start"]) : start_time
		seed_dict["timing_segment_start"] = segment_start

		last_accounted_default = haskey(seed_dict, "last_save_time") ?
			tracking_datetime_or(segment_start, seed_dict["last_save_time"]) : segment_start
		last_accounted = haskey(seed_dict, "timing_last_accounted_time") ?
			tracking_datetime_or(last_accounted_default, seed_dict["timing_last_accounted_time"]) : last_accounted_default
		seed_dict["timing_last_accounted_time"] = last_accounted

		return nothing
	end

	"""
	    register_resume_tracking!(sim_info_path, seed, resume_time, checkpoint_path, dir_path)

	Record in sim.jld2 that `seed` was resumed: status "running", paths, one more resume,
	and a new timing segment starting at `resume_time`. Does nothing if sim.jld2 is missing.
	"""
	function register_resume_tracking!(sim_info_path::String, seed::Int64, resume_time::DateTime,
										  checkpoint_path::String, dir_path::String)
		if !isfile(sim_info_path)
			return
		end

		sim_info = JLD2.load(sim_info_path)
		if !haskey(sim_info, "simulations")
			sim_info["simulations"] = Dict{Any, Any}()
		end
		if !haskey(sim_info["simulations"], seed) || !(sim_info["simulations"][seed] isa AbstractDict)
			sim_info["simulations"][seed] = Dict{String, Any}()
		end

		seed_dict = sim_info["simulations"][seed]
		seed_start_time = haskey(seed_dict, "start_time") ? tracking_datetime_or(resume_time, seed_dict["start_time"]) : resume_time
		seed_dict["start_time"] = seed_start_time
		seed_dict["status"] = "running"
		seed_dict["checkpoint_path"] = checkpoint_path
		seed_dict["dir_path"] = dir_path
		seed_dict["info_path"] = sim_info_path
		ensure_seed_tracking_fields!(seed_dict, seed_start_time)

		seed_dict["timing_resume_count"] = tracking_int_or(0, get(seed_dict, "timing_resume_count", 0)) + 1
		seed_dict["timing_segment_start"] = resume_time
		seed_dict["timing_last_accounted_time"] = resume_time

		JLD2.save(sim_info_path, sim_info)
		return
	end


	"""
	    initialize_sim_tracking(sim_info_path, seed, start_time, checkpoint_path, dir_path, directory)

	Add a fresh tracking entry for `seed` (status "running", paths, zero timing) to an
	existing sim.jld2, creating its `"simulations"` Dict (and sim_num, start time) if needed.
	Does nothing if sim.jld2 does not exist (it is written by the driver).
	"""
	function initialize_sim_tracking(sim_info_path::String, seed::Int64, start_time::DateTime, 
									  checkpoint_path::String, dir_path::String, directory::String)
		if isfile(sim_info_path)
			sim_info = JLD2.load(sim_info_path)
			if !haskey(sim_info, "simulations")
				sim_info["sim_num"] = tryparse(Int, basename(directory))
				sim_info["start_time"] = start_time
				sim_info["info_path"] = sim_info_path
				sim_info["simulations"] = Dict{Any, Any}()
			end
			seed_dict = Dict{String, Any}()
			seed_dict["status"] = "running"
			seed_dict["start_time"] = start_time
			seed_dict["checkpoint_path"] = checkpoint_path
			seed_dict["dir_path"] = dir_path
			seed_dict["info_path"] = sim_info_path
			seed_dict["timing_source"] = "tracked_live"
			seed_dict["timing_active_sec"] = 0.0
			seed_dict["timing_resume_count"] = 0
			seed_dict["timing_segment_start"] = start_time
			seed_dict["timing_last_accounted_time"] = start_time
			seed_dict["timing_mc_step_total_sec"] = 0.0
			seed_dict["timing_mc_step_count"] = 0
			seed_dict["timing_measurement_total_sec"] = 0.0
			seed_dict["timing_measurement_count"] = 0
			sim_info["simulations"][seed] = seed_dict
			JLD2.save(sim_info_path, sim_info)
		end
	end


	"""
	    log_simulation_start(log_filename, sim_data, start_time; continuing=false, checkpoint_path="")

	Append the start of a run to the log: all main parameters, or (`continuing=true`) a
	note that it continues from `checkpoint_path`.
	"""
	function log_simulation_start(log_filename::String, sim_data::SimData, start_time::DateTime; 
								   continuing::Bool=false, checkpoint_path::String="")
		open(log_filename, "a") do io
			if continuing
				println(io, "continuing simulation with existing file: $(checkpoint_path)")
				println(io, "start time: $(start_time)")
			else
				println(io, "----------------------------------------")
				Lx = sim_data.Lx
				Ly = sim_data.Ly
				L_SC = sim_data.SC_L
				U = sim_data.U
				t = sim_data.t
				mu = sim_data.mu
				beta = sim_data.beta
				B = sim_data.B
				num_measure = sim_data.num_measure
				num_thermal = sim_data.num_thermal
				initial_guess = sim_data.guess
				diag_method = sim_data.diag_method
				num_cores = sim_data.num_cores
				TClustersize_x = sim_data.TCluster_size_x
				TClustersize_y = sim_data.TCluster_size_y
				U_disorder = sim_data.U_disorder
				U_puddle_params = sim_data.U_disorder_params
				U_impurity = sim_data.U_impurity
				pairing_symmetry = sim_data.pairing_symmetry
				boundary_conditions = sim_data.boundary_conditions
				uses_u_disorder_params = U_disorder in ["puddles", "on-off-puddles", "gaussian", "impurity-gaussian-kernel"]
				
				println(io, "Simulation parameters: Lx=$(Lx), Ly=$(Ly), L_s=$(L_SC), U=$(U), t=$(t), mu=$(mu), beta=$(beta), B=$(B), num_measure=$(num_measure), num_thermal=$(num_thermal), initial_guess=$(initial_guess), diag_method=$(diag_method), pairing_symmetry=$(pairing_symmetry), num_cores=$(num_cores), $(diag_method in ["TCluster", "K0-TCA"] ? "TClustersize=$(TClustersize_x)x$(TClustersize_y)," : ""), U_disorder=$(U_disorder), $(uses_u_disorder_params ? "U_puddle_params=$(U_puddle_params)," : ""), U_impurity=$(U_impurity), boundary_conditions=$(boundary_conditions)")
				println(io, "start time: $(start_time)")
			end
		end
	end


	"""
	    finalize_simulation_tracking(sim_info_path, seed, log_filename, start_time)

	Mark `seed` as "finished" in sim.jld2 (end time, active time) and log the end of the run.
	"""
	function finalize_simulation_tracking(sim_info_path::String, seed::Int64, 
										   log_filename::String, start_time::DateTime)
		endtime = Dates.now()
		
		# Update sim.jld2
		if isfile(sim_info_path)
			sim_info = JLD2.load(sim_info_path)
			# Ensure simulations dict exists
			if !haskey(sim_info, "simulations")
				sim_info["simulations"] = Dict{Any, Any}()
			end
			if !haskey(sim_info["simulations"], seed)
				sim_info["simulations"][seed] = Dict{String, Any}()
			end
			seed_dict = sim_info["simulations"][seed]
			seed_start_time = haskey(seed_dict, "start_time") ? tracking_datetime_or(start_time, seed_dict["start_time"]) : start_time
			ensure_seed_tracking_fields!(seed_dict, seed_start_time)

			segment_start = tracking_datetime_or(seed_start_time, seed_dict["timing_segment_start"])
			last_accounted = tracking_datetime_or(segment_start, seed_dict["timing_last_accounted_time"])
			base_time = last_accounted >= segment_start ? last_accounted : segment_start
			delta_active_sec = elapsed_seconds(base_time, endtime)
			seed_dict["timing_active_sec"] = tracking_float_or(0.0, get(seed_dict, "timing_active_sec", 0.0)) + delta_active_sec
			seed_dict["timing_last_accounted_time"] = endtime
			seed_dict["end_time"] = endtime
			seed_dict["status"] = "finished"
			sim_info["simulations"][seed] = seed_dict
			JLD2.save(sim_info_path, sim_info)
		end
		
		# Update log file
		open(log_filename, "a") do io
			println(io, "simulation finished!")
			println(io, "end time: $(endtime)")
			println(io, "Elapsed time: $(endtime - start_time)")
			println(io, "----------------------------------------")
		end
	end

end


begin # running the simulation - main functions

	"""
	    running_simulation_once(Lx, Ly, L_SC, U, t, V, seed, mu, beta, B, num_measure, num_thermal,
	        measuring_interval, w_list, initial_guess, U_disorder, diag_method, pairing_symmetry,
	        w_conductivity_range, U_impurity; directory, kwargs...) -> LatticeData

	Run one simulation (one MC seed) at fixed chemical potential μ = `mu`.

	Builds the `SimData` (each argument is a parameter of docs/WORKFLOW.md Appendix A;
	`U_impurity` is `impurity`, `U_puddle_params` is `U_disorder_params`, `U_smooth` is
	`smooth_U`), creates the run folder (`setup_simulation_paths`), then starts a new run,
	or continues from `checkpoint.jld2` if there is one (from scratch if it cannot be read).
	With K0-TCA and a cluster not smaller than the lattice it falls back to K0.
	With `savedata=false` nothing is written and all measurements are kept in memory.
	For runs at fixed filling use `running_simulation_wanted_occupation`
	(occupation_convergence.jl).
	"""
    function running_simulation_once(Lx::Int64, Ly::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, mu::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
		 							directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize_x=4, TClustersize_y=4, U_smooth=(false, 0.8), loop_random=true, gauge_choice="LandauY", boundary_conditions="PBC", delta_values_fixed=Float64[], delta_values_stiffness=Float64[], dw_values=Float64[], compute_jP=false, disorder_seed::Int64=0, snapshot_interval::Int64=500, mu_disorder::String="none", mu_disorder_f::Float64=0.0)

		start_time = Dates.now()
		N = Lx * Ly
		L_min = min(Lx, Ly)
		
		if !savedata
			saving_interval = num_measure # make the measurment arrays full size
		end

		if diag_method=="TCluster" && (TClustersize_x >= Lx || TClustersize_y >= Ly)
			println("Warning: TClustersize should be smaller than L. setting to exact diagonalization")
			TClustersize_x = Lx
			TClustersize_y = Ly
			diag_method = "exact"
		end

		if diag_method=="K0-TCA" && (TClustersize_x >= Lx || TClustersize_y >= Ly)
			println("Warning: TClustersize should be smaller than L. setting to K0")
			TClustersize_x = Lx
			TClustersize_y = Ly
			diag_method = "K0"
		end

        sim_data = SimData(Lx=Lx, Ly=Ly, SC_L=L_SC, U=U, t=t, V=V, seed=seed, mu=mu, beta=beta,
						   num_measure=num_measure, num_thermal=num_thermal, 
						   measuring_interval=measuring_interval, saving_interval=saving_interval,
						   w_list=w_list, guess=initial_guess, U_disorder=U_disorder,
						   diag_method=diag_method, pairing_symmetry=pairing_symmetry,
						   directory=directory, num_cores=num_cores, num_threads=Threads.nthreads(),
						   start_time=start_time, w_conductivity_range=w_conductivity_range, delta_values_fixed=delta_values_fixed, delta_values_stiffness=delta_values_stiffness,
						   U_disorder_params=collect(U_puddle_params), U_impurity=U_impurity,
						   TCluster_size_x=TClustersize_x, TCluster_size_y=TClustersize_y,
						   smooth_U=U_smooth, loop_random=loop_random, B=B,
						   gauge_choice=gauge_choice, boundary_conditions=boundary_conditions,
						   dw_values=dw_values, compute_jP=compute_jP, disorder_seed=disorder_seed,
						   snapshot_interval=snapshot_interval, mu_disorder=mu_disorder, mu_disorder_f=mu_disorder_f)

		# Setup paths
		dir_path, sim_info_path, log_filename, checkpoint_path = setup_simulation_paths(directory, sim_data; date_folder=date_folder, numbered=numbered, seed_folder=seed_folder)
		assert_not_slimmed(dir_path)

		out_Lattice_data = nothing

		if !isfile(checkpoint_path)
			# New simulation
			if savedata
				initialize_sim_tracking(sim_info_path, seed, start_time, checkpoint_path, dir_path, directory)
				log_simulation_start(log_filename, sim_data, start_time; continuing=false)
			end

			out_Lattice_data = MonteCarloSimulation(sim_data, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
		else
			# Continue existing simulation
			if savedata
				register_resume_tracking!(sim_info_path, seed, start_time, checkpoint_path, dir_path)
			end
			log_simulation_start(log_filename, sim_data, start_time; continuing=true, checkpoint_path=checkpoint_path)

			# Try to load existing data, if fails, start from scratch
			try
				data = load_checkpoint(dir_path)
				out_Lattice_data = MonteCarloSimulation(data, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
			catch e
				open(log_filename, "a") do io
					println(io, "Failed to load existing data: $(e). Starting simulation from scratch.")
				end
				out_Lattice_data = MonteCarloSimulation(sim_data, dir_path; saveplots=saveplots, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, sim_info_path=sim_info_path, log_filename=log_filename)
			end
		end

		# Finalize
		if savedata
			finalize_simulation_tracking(sim_info_path, seed, log_filename, start_time)
		end

        return out_Lattice_data
    end

	"""
	    running_simulation_once(L, L_SC, U, ...; TClustersize=4, kwargs...)

	Square-lattice form: `Lx = Ly = L` and a `TClustersize`² cluster.
	"""
	function running_simulation_once(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seed::Int64, mu::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, U_impurity::Float64;
									directory=joinpath(pwd(), "data"), num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=true, numbered=true, seed_folder=false, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize=4, U_smooth=(false, 0.8), loop_random=true, gauge_choice="LandauY", boundary_conditions="PBC", compute_jP=false, disorder_seed::Int64=0, snapshot_interval::Int64=500, mu_disorder::String="none", mu_disorder_f::Float64=0.0)
		# Call rectangular version with Lx = Ly = L
		return running_simulation_once(L, L, L_SC, U, t, V, seed, mu, beta, B, num_measure, num_thermal, measuring_interval, w_list, initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, U_impurity;
									   directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, seed_folder=seed_folder, U_puddle_params=U_puddle_params, TClustersize_x=TClustersize, TClustersize_y=TClustersize, U_smooth=U_smooth, loop_random=loop_random, gauge_choice=gauge_choice, boundary_conditions=boundary_conditions, compute_jP=compute_jP, disorder_seed=disorder_seed, snapshot_interval=snapshot_interval, mu_disorder=mu_disorder, mu_disorder_f=mu_disorder_f)
	end

	"""
	    running_simulation_seeds(Lx, Ly, L_SC, U, t, V, seeds, mu, beta, B, ...; return_data=false, kwargs...)
	    running_simulation_seeds(L, L_SC, U, ...; TClustersize=4, kwargs...)

	`running_simulation_once` for each MC seed in `seeds`, one after another, each in its
	own `seed_<s>` folder; records the end time in `directory/sim.jld2`. With the default
	`disorder_seed = 0` each seed also draws its own disorder; a nonzero `disorder_seed`
	gives all seeds the same disordered sample. Returns the runs if `return_data`.
	"""
	function running_simulation_seeds(Lx::Int64, Ly::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seeds::Vector{Int64}, mu::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, impurity::Float64;
									directory="", num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=false, numbered=false, return_data=false, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize_x=4, TClustersize_y=4, U_smooth=(false, 0.8), loop_random=true, boundary_conditions="PBC", gauge_choice="Mixed", compute_jP=false, disorder_seed::Int64=0, snapshot_interval::Int64=500, mu_disorder::String="none", mu_disorder_f::Float64=0.0)
		out_data = []
		# NOTE: every seed here shares the same `disorder_seed`, i.e. the same
		# disordered sample — these seeds average MC noise, not disorder.
		for seed in seeds
			data = running_simulation_once(Lx, Ly, L_SC, U, t, V, seed, mu, beta, B, num_measure, num_thermal, measuring_interval, w_list, initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, impurity; directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, seed_folder=true, U_puddle_params=U_puddle_params, TClustersize_x=TClustersize_x, TClustersize_y=TClustersize_y, U_smooth=U_smooth, loop_random=loop_random, boundary_conditions=boundary_conditions, gauge_choice=gauge_choice, compute_jP=compute_jP, disorder_seed=disorder_seed, snapshot_interval=snapshot_interval, mu_disorder=mu_disorder, mu_disorder_f=mu_disorder_f)
			if return_data
				push!(out_data, data)
			end
		end
		if savedata && isfile(joinpath(directory, "sim.jld2"))
			sim_info = JLD2.load(joinpath(directory, "sim.jld2"))
			sim_info["endtime"] = Dates.now()
			JLD2.save(joinpath(directory, "sim.jld2"), sim_info)
		end
		return out_data
	end

	# Legacy wrapper for square lattice (L x L)
	function running_simulation_seeds(L::Int64, L_SC::Int64, U::Float64, t::Float64, V::Float64, seeds::Vector{Int64}, mu::Float64, beta::Float64, B::Float64, num_measure::Int64, num_thermal::Int64, measuring_interval::Int64, w_list::Array{Float64,1}, initial_guess::String, U_disorder::String, diag_method::String, pairing_symmetry::String, w_conductivity_range::Array{Float64,1}, impurity::Float64;
									directory="", num_cores=1, savedata=true, saving_interval=500, dump_time=Dates.Hour(1), saveplots=false, date_folder=false, numbered=false, return_data=false, U_puddle_params=(0.0, 6.0, 0.7, 0.7), TClustersize=4, U_smooth=(false, 0.8), loop_random=true, boundary_conditions="PBC", compute_jP=false, disorder_seed::Int64=0, snapshot_interval::Int64=500, mu_disorder::String="none", mu_disorder_f::Float64=0.0)
		# Call rectangular version with Lx = Ly = L
		return running_simulation_seeds(L, L, L_SC, U, t, V, seeds, mu, beta, B, num_measure, num_thermal, measuring_interval, w_list, initial_guess, U_disorder, diag_method, pairing_symmetry, w_conductivity_range, impurity;
										directory=directory, num_cores=num_cores, savedata=savedata, saving_interval=saving_interval, dump_time=dump_time, saveplots=saveplots, date_folder=date_folder, numbered=numbered, return_data=return_data, U_puddle_params=U_puddle_params, TClustersize_x=TClustersize, TClustersize_y=TClustersize, U_smooth=U_smooth, loop_random=loop_random, boundary_conditions=boundary_conditions, compute_jP=compute_jP, disorder_seed=disorder_seed, snapshot_interval=snapshot_interval, mu_disorder=mu_disorder, mu_disorder_f=mu_disorder_f)
	end
end

begin  #computing superfluid stiffness
	# Despite the section name: the single-configuration observables (n, kinetic energy,
	# pair operator, Ps, phase correlations, susceptibilities, LDOS) and the reference
	# current-current / conductivity kernels.

	"""
	    compute_delta_analitic_continuation!(Lattice_data; method="mean_diff_low") -> δ

	Set the broadening δ (`temporary_data.delta`) used to turn the discrete spectrum into
	continuous functions of ω. `"mean_diff_low"`: the mean of the level spacings smaller
	than 2·max(E)/N (spacings < 1e-50 dropped; large gaps, e.g. between Landau levels, are
	left out); falls back to max(E)/N. `"max/N"`: max(E)/N.
	"""
	function compute_delta_analitic_continuation!(Lattice_data::LatticeData; method="mean_diff_low")
		Energies =Lattice_data.current_state.Energies
		N = Lattice_data.sim_data.N

		
		# Sort and compute differences
		sorted_E = sort(vec(Energies))
		max_over_N = maximum(sorted_E) / N
		diffs = diff(sorted_E)
		
		# Filter out numerical noise
		diffs_filtered = filter(x -> x > 1e-50, diffs)
		
		if method == "mean_diff_low" 
			if length(diffs_filtered) > 0
				# Compute max/N threshold
				
				# Filter out large differences (> max/N)
				diffs_low = filter(x -> x < 2*max_over_N, diffs_filtered)
				
				if length(diffs_low) > 0
					# Mean of small differences
					Lattice_data.temporary_data.delta = mean(diffs_low)
				else
					# Fallback to max/N if no small differences found
					Lattice_data.temporary_data.delta = max_over_N
				end
			else
				# Fallback if no differences found
				Lattice_data.temporary_data.delta = max_over_N
			end
			return Lattice_data.temporary_data.delta
		elseif method == "max/N"
			# Directly use max/N
			Lattice_data.temporary_data.delta = max_over_N
			return Lattice_data.temporary_data.delta
		else
			error("Unknown method for compute_delta_analitic_continuation!: $(method)")
		end
		
		
	end


	"""
	    get_FD_dist!(Lattice_data) -> f

	Fermi factors f_l = 1/(1 + e^{βE_l}) of all BdG levels (and e^{βE_l}), stored in
	`temporary_data`.
	"""
	function get_FD_dist!(Lattice_data::LatticeData)
		beta =Lattice_data.sim_data.beta
		Energies = Lattice_data.current_state.Energies
		# Exponential of beta times energies
		exp_beta_Energies = Lattice_data.temporary_data.exp_beta_Energies
		# Fermi function
		f = Lattice_data.temporary_data.f
		@inbounds for l in eachindex(Energies)
			en_l = Energies[l]
			exp_beta_en = exp(beta * en_l)
			f_l = (1 +exp_beta_en)^(-1)
			exp_beta_Energies[l] = exp_beta_en
			f[l] = f_l
		end
		return Lattice_data.temporary_data.f
	end



	"""
	    compute_Kx_operator(data; direction="x", q_vec=nothing) -> Float64

	Kinetic energy per site of the bonds along `direction` ("x" or "y"),
	-(4t/N) Re Σ_i ⟨c†_{i+a} c_i⟩-type sum with the boundary twist and Peierls phase of each
	bond (both spins). With `q_vec`, each site is weighted by e^{-iq·r_i}.
	Its mean enters the stiffness ρs = -¼(⟨Kx⟩ + Λxx(q → 0)).
	"""
	function compute_Kx_operator(data::LatticeData; direction="x", q_vec=nothing)
		U_matrix =data.current_state.U_matrix
		sim_data = data.sim_data
		t = sim_data.t
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		SC_L = sim_data.SC_L
		SC_N = SC_L^2
		f = data.temporary_data.f
		X = data.const_struct.x
		Y = data.const_struct.y
		SC_K_vectors = data.const_struct.SC_K_vectors
		twoN = 2 * N

		if q_vec === nothing
			qx = 0.0
			qy = 0.0
			use_q_phase = false
		elseif length(q_vec) == 2
			qx = Float64(q_vec[1])
			qy = Float64(q_vec[2])
			use_q_phase = !(qx == 0.0 && qy == 0.0)
		else
			error("q_vec must be length 2 when provided")
		end

		if direction == "x"
			new_ix_list = data.const_struct.new_ix_list
			boundary_coords = data.const_struct.x
			L_dir = Lx
			translation_index = 1
		elseif direction == "y"
			new_ix_list = data.const_struct.new_iy_list
			boundary_coords = data.const_struct.y
			L_dir = Ly
			translation_index = 2
		else
			error("Invalid direction for compute_Kx_operator: $(direction). Use 'x' or 'y'.")
		end

		Kx_complex = ComplexF64(0.0)
		for k in 1:SC_N
			Tr_phase = exp(-1im * L_dir * SC_K_vectors[k][translation_index])  # translation phase factor
			for i in 1:N
				FT_phase = use_q_phase ? exp(-1im * (X[i] * qx + Y[i] * qy)) : 1.0
				new_ix = new_ix_list[i]
				Tr = boundary_coords[i] == L_dir ? Tr_phase : 1.0
				Tr *= conj(data.const_struct.peierls_phases[i, translation_index])  # peierls phase factor (-x direction)
				for lambda in 1:twoN
					Kx_complex += Tr * U_matrix[i, lambda, k] * conj(U_matrix[new_ix, lambda, k]) * f[lambda, k] * FT_phase
				end
			end
		end

		Kx = -4*t/(N*SC_N) * real(Kx_complex)

		return Kx
	end

	"""
	    compute_jP_q(data; direction="x", q_vecs=data.const_struct.q_list_CC) -> Vector{ComplexF64}

	Fourier transform of the paramagnetic current density along `direction` in the current
	configuration, for each q in `q_vecs` (note "Kubo formula with magnetic field"):

	    ⟨j^P_a(q)⟩ = -(4t/N) Σ_i e^{-iq·r_i} Σ_λ Im[ conj(P_i,a) U[i,λ] conj(U[i+a,λ]) ] f_λ

	(P = boundary twist × Peierls phase). Same sum as `compute_Kx_operator` with Im instead
	of Re. The convention here is e^{-iq·r}; `fill_FT_table!` uses e^{+iq·r}.
	Measured only when `compute_jP` is on.
	"""
	function compute_jP_q(data::LatticeData; direction::String="x", q_vecs::Array{Array{Float64,1},1}=data.const_struct.q_list_CC)
		U_matrix =data.current_state.U_matrix
		sim_data = data.sim_data
		t = sim_data.t
		Lx = sim_data.Lx
		Ly = sim_data.Ly
		N = sim_data.N
		SC_L = sim_data.SC_L
		SC_N = SC_L^2
		f = data.temporary_data.f
		X = data.const_struct.x
		Y = data.const_struct.y
		SC_K_vectors = data.const_struct.SC_K_vectors
		peierls_phases = data.const_struct.peierls_phases
		twoN = 2 * N
		n_q = length(q_vecs)

		if direction == "x"
			new_ix_list = data.const_struct.new_ix_list
			boundary_coords = data.const_struct.x
			L_dir = Lx
			translation_index = 1
		elseif direction == "y"
			new_ix_list = data.const_struct.new_iy_list
			boundary_coords = data.const_struct.y
			L_dir = Ly
			translation_index = 2
		else
			error("Invalid direction for compute_jP_q: $(direction). Use 'x' or 'y'.")
		end

		jP = zeros(ComplexF64, n_q)
		for k in 1:SC_N
			Tr_phase = exp(-1im * L_dir * SC_K_vectors[k][translation_index])
			for i in 1:N
				new_ix = new_ix_list[i]
				Tr = boundary_coords[i] == L_dir ? Tr_phase : (1.0 + 0.0im)
				Tr *= conj(peierls_phases[i, translation_index])
				local_im_sum = 0.0
				for lambda in 1:twoN
					local_im_sum += imag(Tr * U_matrix[i, lambda, k] * conj(U_matrix[new_ix, lambda, k])) * f[lambda, k]
				end
				for iq in 1:n_q
					q = q_vecs[iq]
					jP[iq] += exp(-1im * (q[1]*X[i] + q[2]*Y[i])) * local_im_sum
				end
			end
		end

		jP .*= -4*t / (N * SC_N)

		return jP
	end


	"""
	    g_vec(Lattice_data, i, si, j, sj; g=:g_A) -> Vector{ComplexF64}

	g[l] = conj(U[i_si, l]) U[j_sj, l] f_l for BdG rows i_si = i (si = +1) or i + N
	(si = -1), written into the scratch vector `g_A` or `g_B`. Helper of the old
	`compute_D_dagger_D`.
	"""
	function g_vec(Lattice_data::LatticeData, i::Int64, si::Int64, j::Int64, sj::Int64; g=:g_A)
		U_matrix =Lattice_data.current_state.U_matrix
		f = Lattice_data.temporary_data.f
		N = Lattice_data.sim_data.N

		if g == :g_A
			g_out = Lattice_data.temporary_data.g_A
		elseif g==:g_B
			g_out = Lattice_data.temporary_data.g_B
		end
		U_vec_i = view(U_matrix, i + Int((1 - si) * N / 2), :)
		U_vec_j = view(U_matrix, j + Int((1 - sj) * N / 2), :)
		

		for l in 1:(2 * N)
			s = conj(U_vec_i[l]) * U_vec_j[l] * f[l]
			g_out[l] = s 
		end
		return g_out
	end

	"""
	    avrage_n(Lattice_data) -> Float64

	Filling n = (2/N) Σ_i Σ_λ |U[i, λ]|² f_λ of the current configuration (K = 0 only).
	The measurement uses `avrage_n_K`.
	"""
	function avrage_n(Lattice_data::LatticeData)
		U_matrix =Lattice_data.current_state.U_matrix
		sim_data = Lattice_data.sim_data
		N = sim_data.N
		f = Lattice_data.temporary_data.f
		n=0.0

		for i in 1:N
			for lambda in 1:2*N
				n += 2 * abs2(U_matrix[i, lambda]) * f[lambda] / N
			end
			
		end
		return n
	end

	"""
	    avrage_n_K(Lattice_data) -> Float64

	Filling n (electrons per site, 0 to 2) of the current configuration, summed over all
	supercell sectors K: n = 2/(N·SC_N) Σ_K Σ_λ Σ_i |U[i, λ, K]|² f_λ,K.
	"""
	function avrage_n_K(Lattice_data::LatticeData)
		sim_data =Lattice_data.sim_data
		N = sim_data.N
		L_SC = sim_data.SC_L
		N_SC = L_SC^2
		U_matrix = Lattice_data.current_state.U_matrix
		f = Lattice_data.temporary_data.f
		n=0.0

		for k in 1:N_SC
			for lambda in 1:2*N
				U2 = 0.0
				for i in 1:N
					@inbounds U2 += abs2(U_matrix[i, lambda, k])
				end
				@inbounds n += U2 * f[lambda, k]
			end
		end
		n *= 2/(N * N_SC)

		return n
	end

	"""
	    staggered_density_K(Lattice_data) -> Float64

	Charge-density-wave order parameter m_CDW = (1/N) Σ_i cos(Q·r_i) ⟨n_i⟩ (Q = `Q_density`).
	Same sum as `avrage_n_K` with each site weighted by cos(Q·r_i); for Q = 0 it is just n,
	so it is measured only for Q ≠ 0.
	"""
	function staggered_density_K(Lattice_data::LatticeData)
		sim_data =Lattice_data.sim_data
		N = sim_data.N
		L_SC = sim_data.SC_L
		N_SC = L_SC^2
		U_matrix = Lattice_data.current_state.U_matrix
		f = Lattice_data.temporary_data.f
		dsign = Lattice_data.const_struct.density_sign
		m = 0.0

		for k in 1:N_SC
			for lambda in 1:2*N
				w = 0.0
				for i in 1:N
					@inbounds w += dsign[i] * abs2(U_matrix[i, lambda, k])
				end
				@inbounds m += w * f[lambda, k]
			end
		end
		m *= 2/(N * N_SC)

		return m
	end

	"""
	    compute_Delta_operator(data) -> ComplexF64

	Mean pair amplitude of the electrons, ⟨Δ̂⟩ = (1/N) Σ_i ⟨c_i↓ c_i↑⟩
	= (1/N) Σ_{i,λ} U[i, λ] conj(U[i+N, λ]) f_λ (K = 0 sector only).
	Compare with the mean of the auxiliary field Δ_i (`Delta_mean`).
	"""
	function compute_Delta_operator(data::LatticeData)
		U_matrix =data.current_state.U_matrix
		sim_data = data.sim_data
		N = sim_data.N
		f = data.temporary_data.f
		K_ix = 1

		Delta_op = ComplexF64(0.0)
		for lambda in 1:2N
			U_mult = ComplexF64(0.0)
			for i in 1:N
				U_mult += U_matrix[i, lambda, K_ix] * conj(U_matrix[i + N, lambda, K_ix]) 
			end
			Delta_op += U_mult * f[lambda, K_ix]
		end
		Delta_op /= N
		return Delta_op
	end


	"""
	    compute_D_dagger_D(Lattice_Data)

	Old ⟨Δ†Δ⟩ estimate. Broken (uses an undefined `Delta_operator`) and not used by the
	pipeline; see `compute_D_dagger_D_K` / `compute_D_dagger_D_fast!`.
	"""
	function compute_D_dagger_D(Lattice_Data::LatticeData)
		sim_data =Lattice_Data.sim_data
		N = sim_data.N
		exp_beta_Energies = Lattice_Data.temporary_data.exp_beta_Energies


		DdD = Float64(0.0)
		
		# Compute Ps using the formula
		for i in 1:N
			for j in 1:N
				g_alpha = g_vec(Lattice_Data, i, 1, j, 1; g=:g_A)
				g_alpha_sum = sum(g_alpha)
				g_beta = g_vec(Lattice_Data, j, -1, i, -1; g=:g_B)
				g_beta_sum = dot(exp_beta_Energies, g_beta)
				DdD += (g_alpha_sum * g_beta_sum) / N
			end
		end
		DdD += abs2(Delta_operator)
		return real(DdD)
	end

	"""
	    compute_D_dagger_D_V2(Lattice_data) -> Float64

	⟨Δ†Δ⟩/N by a direct O(N⁴) Wick sum over sites (K = 0). Not used by the pipeline; kept
	as a slow cross-check of `compute_D_dagger_D_K`.
	"""
	function compute_D_dagger_D_V2(Lattice_data::LatticeData)
		N = Lattice_data.sim_data.N
		f = Lattice_data.temporary_data.f
		U_matrix = Lattice_data.current_state.U_matrix
		DdD = Float64(0.0)
		for i in 1:N
			for j in 1:N
				for l in 1:2N
					mul_term = 0.0
					a = U_matrix[j, l]
					b = U_matrix[i + N, l]
					for k in 1:2N
						c = U_matrix[i + N, k] * a
						# = conj(U[j+N,k]) (U[i+N,l] U[j,k] f_k + U[i+N,k] U[j,l] (1 - f_k))
						mul_term += conj(U_matrix[j + N, k]) * ((b * U_matrix[j, k]  - c) * f[k] + c)
					end
					DdD += f[l] * conj(U_matrix[i, l]) * mul_term
				end
			end
		end
		DdD = real(DdD / N)
		return DdD
	end


	"""
	    compute_D_dagger_D_K(data) -> Float64

	⟨Δ†Δ⟩/N of the pair operator Δ = Σ_i c_i↓ c_i↑, summed over supercell sectors:
	(Σ_lm f_l (1 - f_m) |V_lm|² + |Σ_l f_l V_ll|²)/(N·SC_N) with V from `compute_V_k`.
	Reference kernel of `compute_D_dagger_D_fast!`; gives Ps = 2·DdD - n + 1.
	"""
	function compute_D_dagger_D_K(data::LatticeData)
		N =data.sim_data.N
		L_SC = data.sim_data.SC_L
		N_SC = L_SC^2
		f = data.temporary_data.f

		DdD = Float64(0.0)
		add2_term = ComplexF64(0.0)

		for K in 1:N_SC
			for l in 1:2N
				for m in 1:2N
					V_k = compute_V_k(data, K, l, m)
					if l==m
						add2_term += f[l, K] * V_k
					end
					DdD += f[l, K] * (1 - f[m, K]) * abs2(V_k)
				end
			end
		end
		DdD += abs2(add2_term)
		DdD /= (N * N_SC)
		return DdD
	end

	"""
	    compute_V_k(Lattice_data, k, l, m) -> ComplexF64

	V_lm = Σ_r conj(U[r, l, k]) U[r+N, m, k]: overlap of the up part of level l with the
	down part of level m in sector k (one entry of U_up' U_dn).
	"""
	function compute_V_k(Lattice_data::LatticeData, k::Int64, l::Int64, m::Int64)
		N =Lattice_data.sim_data.N

		U_matrix = Lattice_data.current_state.U_matrix
		V_k = zero(ComplexF64)
		for r in 1:N
			V_k += conj(U_matrix[r, l, k]) * U_matrix[r + N, m, k]
		end
		return V_k
	end

	"""
	    compute_Ps_V2(Lattice_data; DdD=nothing, n=nothing)

	Ps = 2·DdD + 1 - n. Not used by the pipeline (it calls the broken
	`compute_D_dagger_D` if `DdD` is not given); `get_means` computes Ps from the streams.
	"""
	function compute_Ps_V2(Lattice_data::LatticeData; DdD=nothing, n=nothing)
		if !(typeof(DdD) <: Number)
			DdD = compute_D_dagger_D(Lattice_data)
		end
		if !(typeof(n) <: Number)
			n = avrage_n(Lattice_data)
		end
		return 2*DdD + 1 - n
	end

	"""
	    compute_PS_from_Ps_l(Lattice_data, Ps_l; n=nothing)

	Ps = 2 Σ_l Ps_l / N + n - 1 from a pair correlation over all N separations.
	Not used by the pipeline.
	"""
	function compute_PS_from_Ps_l(Lattice_data::LatticeData, Ps_l::Array{Float64,1}; n=nothing)
		N = Lattice_data.sim_data.N
		if length(Ps_l) != N
			throw(ArgumentError("Ps_l must be a vector of length N"))
		end
		if !(typeof(n) <: Number)
			n = avrage_n(Lattice_data)
		end
		Ps = 2*sum(Ps_l)/N + n -1
		return Ps
	end

	"""
	    compute_DdD_from_Ps_l(Lattice_data, Ps_l; n=nothing)

	DdD = Σ_l Ps_l / N + n - 1 from a pair correlation over all N separations.
	Not used by the pipeline.
	"""
	function compute_DdD_from_Ps_l(Lattice_data::LatticeData, Ps_l::Array{Float64,1}; n=nothing)
		N = Lattice_data.sim_data.N
		if length(Ps_l) != N
			throw(ArgumentError("Ps_l must be a vector of length N"))
		end
		if !(typeof(n) <: Number)
			n = avrage_n(Lattice_data)
		end
		DdD = sum(Ps_l)/N + n - 1
		return DdD
	end

	"""
	    compute_Ps_l(Lattice_data, l) -> Float64

	Old O(N³) version of the pair correlation at separation `l = [lx, ly]` (K = 0).
	Not used by the pipeline; the measurement uses `compute_Ps_l_K`.
	"""
	function compute_Ps_l(Lattice_data::LatticeData, l::Array{Int64,1})
		if length(l) != 2
			throw(ArgumentError("l must be a vector of length 2"))
		end
		f = Lattice_data.temporary_data.f
		U_matrix = Lattice_data.current_state.U_matrix
		Lx = Lattice_data.sim_data.Lx
		Ly = Lattice_data.sim_data.Ly
		N = Lattice_data.sim_data.N
		CI = Lattice_data.const_struct.CI
		Ps = 0.0
		for i in 1:N
			i_x = get_x(i, CI)
			i_y = get_y(i, CI)
			l_ix = idx(add_xy_ix(i_x, l[1], Lx), add_xy_ix(i_y, l[2], Ly), Lx)
			for m in 1:2N
				m_out = f[m] * conj(U_matrix[i + N, m, 1])
				m1_term = m_out * U_matrix[i, m, 1]
				m2_term = m_out * U_matrix[l_ix + N, m, 1]
				for k in 1:2N
					fk = f[k]
					Ps += conj(U_matrix[l_ix, k, 1]) * (m1_term  *  U_matrix[l_ix + N, k, 1] * fk + m2_term * U_matrix[i, k, 1] * (1 - fk))
				end
			end
		end
		return real(Ps)
	end

	"""
	    compute_Ps_l_K(data, l) -> Float64

	Equal-time s-wave pair correlation at separation `l = [lx, ly]` (the `Ps_l` measurement,
	over the separations in `ls_list`), as in Scalettar, Trivedi & Huscroft, PRB 1999, Eq. 9:
	P_s(l) = ⟨Δ_i Δ†_{i+l}⟩ with Δ†_i = c†_i↑ c†_i↓.

	By Wick's theorem it is Σ_i [ F_i^* F_{i+l} + (G↓_{i+l,i})² ] / SC_N, with
	F_i = Σ_m conj(U[i+N,m]) U[i,m] f_m the local pair amplitude and G↓ the down-spin
	propagator from i to i + l. It is summed over the sites i (N × the site average), so
	that its sum over all separations l gives ⟨Δ̂†Δ̂⟩, and with it the pair-field structure
	factor Ps = 2⟨Δ̂†Δ̂⟩ + 1 - n (thesis Appendix C). At large l it tends to Δ_OP².
	"""
	function compute_Ps_l_K(data::LatticeData, l::Array{Int64,1})
		# compute Ps_l using supercell K vector basis
		# (validated for SC_L = 1; the supercell sum over K is not checked)
		if length(l) != 2
			throw(ArgumentError("l must be a vector of length 2"))
		end
		f = data.temporary_data.f
		U_matrix = data.current_state.U_matrix
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		N = data.sim_data.N
		L_SC = data.sim_data.SC_L
		N_SC = L_SC^2

		Ps = ComplexF64(0.0)
		

		for i in 1:N
			i_x = data.const_struct.x[i]
			i_y = data.const_struct.y[i]
			l_ix = idx(add_xy_ix(i_x, l[1], Lx), add_xy_ix(i_y, l[2], Ly), Lx)
			L_vec = [Lx * div(i_x + l[1] - 1, Lx), Ly * div(i_y + l[2] - 1, Ly)]
			add2_term = ComplexF64(0.0)
			G_r_sum_conj = ComplexF64(0.0)
			G_l_sum = ComplexF64(0.0)

			for K in 1:N_SC
				LK_phase = exp(1im * dot(L_vec, data.const_struct.SC_K_vectors[K]))
				G_rl_dd = ComplexF64(0.0)

				for m in 1:2N
					G_r_sum_conj += conj(U_matrix[i + N, m, K]) * U_matrix[i, m, K] * f[m, K]
					G_l_sum += conj(U_matrix[l_ix , m, K]) * U_matrix[l_ix + N, m, K] * f[m, K]
					G_rl_dd += conj(U_matrix[l_ix + N, m, K]) * U_matrix[i + N, m, K] * f[m, K]
				end

				add2_term += LK_phase * G_rl_dd
			end
			Ps += G_r_sum_conj * G_l_sum + add2_term^2
		end
		Ps /= N_SC
		return real(Ps)
	end




	"""
	    current_correlation_q_vec(Lattice_data, q_vecs; direction1="x", direction2="x") -> Vector

	Static current-current correlation Λ_ab(q, ω = 0) for each q in `q_vecs`
	(a = `direction1`, b = `direction2`; real for a = b). K = 0 sector only.
	Not used by the measurement (which calls `measure_current_correlations_q!` or the fast kernel).
	"""
	function current_correlation_q_vec(Lattice_data::LatticeData, q_vecs::Array{Array{Float64,1},1}; direction1::String="x", direction2::String="x")
		# Compute G matrices for the specified directions
		if direction1 == direction2
			G_vec_1, _ = make_G_matrix_q_vec(Lattice_data, q_vecs; G_out=Lattice_data.temporary_data.G_matrix_vec_x, direction=direction1)
			G_vec_2 = G_vec_1
		else
			G_vec_1, _ = make_G_matrix_q_vec(Lattice_data, q_vecs; G_out=Lattice_data.temporary_data.G_matrix_vec_x, direction=direction1)
			G_vec_2, _ = make_G_matrix_q_vec(Lattice_data, q_vecs; G_out=Lattice_data.temporary_data.G_matrix_vec_y, direction=direction2)
		end
		
		# Use the helper function to compute the correlation
		Lxx_vec = Lattice_data.temporary_data.Lxx_temp
		current_correlation_from_G!(Lattice_data, G_vec_1, G_vec_2, Lxx_vec; q_length=length(q_vecs))

		if direction1 == direction2
			Lxx_vec = real.(Lxx_vec)
		end
		return Lxx_vec
	end


	"""
	    current_correlation_from_G!(Lattice_data, G_vec_1, G_vec_2, Lxx_temp; q_length=length(G_vec_1)) -> Lxx_temp

	Static current-current correlation from precomputed current vertices (one matrix per q):

	    Λ_ab(q) = (t²/N) [ -Σ_lm G^a_lm conj(G^b_lm) M_lm + β (Σ_l G^a_ll f_l) conj(Σ_l G^b_ll f_l) ]

	with M_lm = (f_l - f_m)/(E_l - E_m) (β f_l (f_l - 1) for equal energies).
	Written into `Lxx_temp` (first `q_length` entries). Reference for `current_correlation_q_fast!`.
	"""
	function current_correlation_from_G!(Lattice_data::LatticeData, G_vec_1::Array{Array{ComplexF64,2},1}, G_vec_2::Array{Array{ComplexF64,2},1}, Lxx_temp::Array{ComplexF64,1}; q_length=length(G_vec_1))
		# Extract simulation parameters
		Energies = Lattice_data.current_state.Energies
		f = Lattice_data.temporary_data.f
		beta = Lattice_data.sim_data.beta
		t = Lattice_data.sim_data.t
		N = Lattice_data.sim_data.N
		twoN = 2 * N
		eps_diff = 1e-12
		prefactor = t^2 / N
		if length(Lxx_temp) < q_length
			Lxx_temp = zeros(ComplexF64, q_length)
		end
		
		# Compute the current-current correlation for each q
		@inbounds for ix in 1:q_length
			G1 = G_vec_1[ix]
			G2 = G_vec_2[ix]
			Lxx_temp[ix] = ComplexF64(0.0)
			LL2_add_1 = ComplexF64(0.0)
			LL2_add_2 = ComplexF64(0.0)
		
			for l in 1:twoN
				f_l = f[l]
				LL2_add_1 += G1[l,l] * f_l
				LL2_add_2 += G2[l,l] * f_l
				
				for m in 1:twoN
					diff = Energies[l] - Energies[m]
					if abs(diff) < eps_diff
						Mlm = beta * f_l * (f_l - 1)
					else
						Mlm = (f_l - f[m]) / diff
					end
					Lxx_temp[ix] -= G1[l,m] * conj(G2[l,m]) * Mlm
				end
			end

			Lxx_temp[ix] += beta * LL2_add_1 * conj(LL2_add_2)
			Lxx_temp[ix] *= prefactor
		end
		
		return Lxx_temp
	end

	"""
	    current_correlation_from_G_delta!(Lattice_data, G_vec_1, G_vec_2, Lxx_temp; q_length, delta=0.0)

	Same as `current_correlation_from_G!` with M_lm regularized by a broadening `delta`:
	(f_l - f_m)(ΔE - iδ)/(ΔE² + δ²). Not used by the pipeline.
	"""
	function current_correlation_from_G_delta!(Lattice_data::LatticeData, G_vec_1::Array{Array{ComplexF64,2},1}, G_vec_2::Array{Array{ComplexF64,2},1}, Lxx_temp::Array{ComplexF64,1};
											 q_length=length(G_vec_1), delta::Float64=0.0)
		# Extract simulation parameters
		Energies = Lattice_data.current_state.Energies
		f = Lattice_data.temporary_data.f
		beta = Lattice_data.sim_data.beta
		t = Lattice_data.sim_data.t
		N = Lattice_data.sim_data.N
		if length(Lxx_temp) < q_length
			Lxx_temp = zeros(ComplexF64, q_length)
		end
		
		# Compute the current-current correlation for each q
		for ix in 1:q_length
			G1 = G_vec_1[ix]
			G2 = G_vec_2[ix]
			Lxx_temp[ix] = ComplexF64(0.0)
			LL2_add_1 = ComplexF64(0.0)
			LL2_add_2 = ComplexF64(0.0)
		
			for l in 1:2N
				LL2_add_1 += G1[l,l] * f[l]
				LL2_add_2 += G2[l,l] * f[l]
				
				for m in 1:2N
					diff = Energies[l] - Energies[m]
					Mlm = 0.0
					if diff ≈ 0 && delta == 0.0
						Mlm = beta * f[l] * (f[l] - 1)
					else
						Mlm = (f[l] - f[m]) * (diff - 1im * delta) * ((diff)^2 + delta^2)^(-1)
					end
					
					Lxx_temp[ix] -= G1[l,m] * conj(G2[l,m]) * Mlm
				end
			end

			Lxx_temp[ix] += beta * LL2_add_1 * conj(LL2_add_2)
			Lxx_temp[ix] *= t^2 / N
		end
		
		return Lxx_temp
	end


	"""
	    measure_current_correlations_q!(data, measurements, i; return_Gs=false)

	Reference kernel of the current-correlation measurement: build the current vertices
	G^x(q), G^y(q) for `q_list_CC` (q = 0, (0, qy), (qx, 0)) and store Λxx, Λxy, Λyx, Λyy at
	these q in measurement `i`. Returns the vertices if `return_Gs`.
	"""
	function measure_current_correlations_q!(data::LatticeData, measurements::MeasureData, i::Int; return_Gs=false)
		q_list_CC =data.const_struct.q_list_CC
		L_gen_y = data.sim_data.Ly * data.sim_data.SC_L
		L_gen_x = data.sim_data.Lx * data.sim_data.SC_L
		n_qy = div(L_gen_y, 2)
		n_qx = div(L_gen_x, 2)
		qy_start = 2
		qy_end = 1 + n_qy
		qx_start = qy_end + 1
		qx_end = qy_end + n_qx
		Lxx_temp = data.temporary_data.Lxx_temp
		
		# Precompute G matrices for x and y directions (done once for all 4 components)
		G_x_vec, _ = make_G_matrix_q_vec(data, q_list_CC; G_out=data.temporary_data.G_matrix_vec_x, direction="x")
		G_y_vec, _ = make_G_matrix_q_vec(data, q_list_CC; G_out=data.temporary_data.G_matrix_vec_y, direction="y")
		
		# Lxx (diagonal - result is real). Index 1 of Lxx_temp is q=0, w=0.
		current_correlation_from_G!(data, G_x_vec, G_x_vec, Lxx_temp; q_length=length(q_list_CC))
		measurements.Lxx_q0[i] = real(Lxx_temp[1])
		measurements.Lxx_qy[i] .= real.(Lxx_temp[qy_start:qy_end])
		measurements.Lxx_qx[i] .= real.(Lxx_temp[qx_start:qx_end])

		# Lxy (off-diagonal - result is complex)
		current_correlation_from_G!(data, G_x_vec, G_y_vec, Lxx_temp; q_length=length(q_list_CC))
		measurements.Lxy_q0[i] = Lxx_temp[1]
		measurements.Lxy_qy[i] .= Lxx_temp[qy_start:qy_end]
		measurements.Lxy_qx[i] .= Lxx_temp[qx_start:qx_end]

		# Lyx (off-diagonal - result is complex)
		current_correlation_from_G!(data, G_y_vec, G_x_vec, Lxx_temp; q_length=length(q_list_CC))
		measurements.Lyx_q0[i] = Lxx_temp[1]
		measurements.Lyx_qy[i] .= Lxx_temp[qy_start:qy_end]
		measurements.Lyx_qx[i] .= Lxx_temp[qx_start:qx_end]

		# Lyy (diagonal - result is real)
		current_correlation_from_G!(data, G_y_vec, G_y_vec, Lxx_temp; q_length=length(q_list_CC))
		measurements.Lyy_q0[i] = real(Lxx_temp[1])
		measurements.Lyy_qy[i] .= real.(Lxx_temp[qy_start:qy_end])
		measurements.Lyy_qx[i] .= real.(Lxx_temp[qx_start:qx_end])
		
		if return_Gs
			return G_x_vec, G_y_vec
		end
		return nothing
	end

	# Current vertex G^a(q)[l,m] = ⟨l| j^a(q) |m⟩ in the BdG eigenbasis. In real space it
	# is G_r(i) = S_up(l,m)(i) - S_down(m,l)(i), the bond current from site i to i + a in
	# the pair of levels (l, m); G(q) = Σ_i e^{iq·r_i} G_r(i).

	"""
	    build_G_r_from_S_r_ref!(Lattice_data, G_r, l, m, K; direction="x") -> G_r

	Reference version of `build_G_r_from_S_r!` built from two `make_S_r` calls; kept for
	equivalence checks (`validate_build_G_r_direct!`).
	"""
	function build_G_r_from_S_r_ref!(Lattice_data::LatticeData, G_r::Array{ComplexF64,1}, l::Int64, m::Int64, K::Int64; direction::String="x")
		S_up_r = make_S_r(Lattice_data, l, m, 1, K; direction=direction)
		@inbounds @simd for ix in eachindex(G_r)
			G_r[ix] = S_up_r[ix]
		end

		# Down term uses swapped (m,l) indices to match the legacy transpose(S_down) convention.
		S_dn_swapped_r = make_S_r(Lattice_data, m, l, -1, K; direction=direction)
		@inbounds @simd for ix in eachindex(G_r)
			G_r[ix] -= S_dn_swapped_r[ix]
		end

		return G_r
	end

	"""
	    build_G_r_from_S_r_x!(Lattice_data, G_r, l, m, K) -> G_r

	Real-space current vertex along +x for levels (l, m) in sector K:
	G_r(i) = S_up(l,m)(i) - S_down(m,l)(i), one pass over the sites, with the boundary twist
	and Peierls phase of each bond.
	"""
	function build_G_r_from_S_r_x!(Lattice_data::LatticeData, G_r::Array{ComplexF64,1}, l::Int64, m::Int64, K::Int64)
		Lx = Lattice_data.sim_data.Lx
		N = Lattice_data.sim_data.N
		U_matrix = Lattice_data.current_state.U_matrix
		const_struct = Lattice_data.const_struct

		phase_Tr = exp(-1im * 2π/Lx * const_struct.SC_K_vectors[K][1])
		new_ix_list = const_struct.new_ix_list
		x_list = const_struct.x
		peierls_phases = const_struct.peierls_phases

		@inbounds for ix in 1:N
			new_ix = new_ix_list[ix]
			Tr = (x_list[ix] == Lx ? phase_Tr : (1.0 + 0.0im)) * peierls_phases[ix, 1]

			# S_up(l,m)
			S_up = conj(Tr) * (conj(U_matrix[new_ix, l, K]) * U_matrix[ix, m, K]) - Tr * (conj(U_matrix[ix, l, K]) * U_matrix[new_ix, m, K])

			# S_down_swapped(m,l) = make_S_r(m,l,spin=-1)
			ix_dn = ix + N
			new_ix_dn = new_ix + N
			S_dn_swapped = conj(Tr) * (U_matrix[new_ix_dn, m, K] * conj(U_matrix[ix_dn, l, K])) - Tr * (U_matrix[ix_dn, m, K] * conj(U_matrix[new_ix_dn, l, K]))

			G_r[ix] = S_up - S_dn_swapped
		end

		return G_r
	end

	"""
	    build_G_r_from_S_r_y!(Lattice_data, G_r, l, m, K) -> G_r

	Same as `build_G_r_from_S_r_x!` for bonds along +y.
	"""
	function build_G_r_from_S_r_y!(Lattice_data::LatticeData, G_r::Array{ComplexF64,1}, l::Int64, m::Int64, K::Int64)
		Ly = Lattice_data.sim_data.Ly
		N = Lattice_data.sim_data.N
		U_matrix = Lattice_data.current_state.U_matrix
		const_struct = Lattice_data.const_struct

		phase_Tr = exp(-1im * 2π/Ly * const_struct.SC_K_vectors[K][2])
		new_ix_list = const_struct.new_iy_list
		y_list = const_struct.y
		peierls_phases = const_struct.peierls_phases

		@inbounds for ix in 1:N
			new_ix = new_ix_list[ix]
			Tr = (y_list[ix] == Ly ? phase_Tr : (1.0 + 0.0im)) * peierls_phases[ix, 2]

			# S_up(l,m)
			S_up = conj(Tr) * (conj(U_matrix[new_ix, l, K]) * U_matrix[ix, m, K]) - Tr * (conj(U_matrix[ix, l, K]) * U_matrix[new_ix, m, K])

			# S_down_swapped(m,l) = make_S_r(m,l,spin=-1)
			ix_dn = ix + N
			new_ix_dn = new_ix + N
			S_dn_swapped = conj(Tr) * (U_matrix[new_ix_dn, m, K] * conj(U_matrix[ix_dn, l, K])) - Tr * (U_matrix[ix_dn, m, K] * conj(U_matrix[new_ix_dn, l, K]))

			G_r[ix] = S_up - S_dn_swapped
		end

		return G_r
	end

	"""
	    build_G_r_from_S_r!(Lattice_data, G_r, l, m, K; direction="x")

	`build_G_r_from_S_r_x!` or `_y!` by `direction`.
	"""
	function build_G_r_from_S_r!(Lattice_data::LatticeData, G_r::Array{ComplexF64,1}, l::Int64, m::Int64, K::Int64; direction::String="x")
		if direction == "x"
			return build_G_r_from_S_r_x!(Lattice_data, G_r, l, m, K)
		elseif direction == "y"
			return build_G_r_from_S_r_y!(Lattice_data, G_r, l, m, K)
		else
			error("direction must be 'x' or 'y'")
		end
	end

	"""
	    validate_build_G_r_direct!(Lattice_data; nsamples=12, atol=1e-12, rtol=1e-10, seed=1234)
	        -> (ok, max_abs_diff, worst_case)

	Compare `build_G_r_from_S_r!` with the reference `build_G_r_from_S_r_ref!` on
	`nsamples` random level pairs in both directions. A check, not used by the pipeline.
	"""
	function validate_build_G_r_direct!(Lattice_data::LatticeData; nsamples::Int=12, atol::Float64=1e-12, rtol::Float64=1e-10, seed::Int=1234)
		Lx = Lattice_data.sim_data.Lx
		Ly = Lattice_data.sim_data.Ly
		if size(Lattice_data.temporary_data.S_r, 1) != Lx || size(Lattice_data.temporary_data.S_r, 2) != Ly
			Lattice_data.temporary_data.S_r = zeros(ComplexF64, Lx, Ly)
			Lattice_data.temporary_data.plan_S_r = plan_fft(Lattice_data.temporary_data.S_r)
		end

		N = Lattice_data.sim_data.N
		twoN = 2 * N
		SC_N = Lattice_data.sim_data.SC_L^2
		rng = Random.MersenneTwister(seed)

		G_ref = zeros(ComplexF64, N)
		G_new = zeros(ComplexF64, N)
		max_abs_diff = 0.0
		worst_case = (l=1, m=1, K=1, direction="x")

		for _ in 1:nsamples
			l = rand(rng, 1:twoN)
			m = rand(rng, 1:twoN)
			K = rand(rng, 1:SC_N)
			for direction in ("x", "y")
				build_G_r_from_S_r_ref!(Lattice_data, G_ref, l, m, K; direction=direction)
				build_G_r_from_S_r!(Lattice_data, G_new, l, m, K; direction=direction)
				d = maximum(abs.(G_new .- G_ref))
				if d > max_abs_diff
					max_abs_diff = d
					worst_case = (l=l, m=m, K=K, direction=direction)
				end
			end
		end

		ok = isapprox(max_abs_diff, 0.0; atol=atol, rtol=rtol)
		return (ok=ok, max_abs_diff=max_abs_diff, worst_case=worst_case)
	end

	"""
	    get_FT_table_for_q_vecs!(Lattice_data, q_vecs) -> Matrix{ComplexF64}

	Fourier table FT[i, q] = e^{iq·r_i} for `q_vecs`: the precomputed table if `q_vecs` is
	the current-correlation or susceptibility q list (matched by `q_list_signature`),
	else computed into reusable scratch.
	"""
	function get_FT_table_for_q_vecs!(Lattice_data::LatticeData, q_vecs::Array{Array{Float64,1},1})
		const_data = Lattice_data.const_struct
		q_length = length(q_vecs)
		sig = q_list_signature(q_vecs)

		if q_length == size(const_data.FT_vecs_q_CC, 2) && sig == const_data.q_list_CC_sig
			return const_data.FT_vecs_q_CC
		elseif q_length == size(const_data.FT_vecs_q_sus, 2) && sig == const_data.q_list_sus_sig
			return const_data.FT_vecs_q_sus
		end

		# Fallback for custom/changed q vectors: compute FT on-the-spot into reusable temp storage.
		FT_vecs = Lattice_data.temporary_data.FT_vecs_q
		N = Lattice_data.sim_data.N
		if size(FT_vecs, 2) < q_length
			FT_vecs = zeros(ComplexF64, N, q_length)
			Lattice_data.temporary_data.FT_vecs_q = FT_vecs
		end
		fill_FT_table!(FT_vecs, q_vecs, const_data.x, const_data.y)
		return FT_vecs
	end


	"""
	    make_G_matrix_q_vec(Lattice_data, q_vecs; G_out=nothing, direction="x") -> (G_out, nothing)

	Reference current vertices G^a(q) (2N×2N, one per q in `q_vecs`, K = 0), built entry by
	entry from `build_G_r_from_S_r_*` and the Fourier table. Written into `G_out` (default:
	the `G_matrix_vec_x/_y` buffers). The fast kernel is `current_vertices_fast!`.
	"""
	function make_G_matrix_q_vec(Lattice_data::LatticeData, q_vecs::Array{Array{Float64,1},1};
		G_out=nothing, direction::String="x")
		N = Lattice_data.sim_data.N
		twoN = 2 * N
		q_length = length(q_vecs)
		ensure_G_buffers!(Lattice_data) # G_matrix_vec_x/_y are allocated on first use
		FT_vecs = get_FT_table_for_q_vecs!(Lattice_data, q_vecs)
		G_r = Lattice_data.temporary_data.G_r_vec
		q_proj = Lattice_data.temporary_data.G_q_proj
		# Choose preallocation for G_out
		if G_out === nothing
			if direction == "x"
				G_out = Lattice_data.temporary_data.G_matrix_vec_x
			elseif direction == "y"
				G_out = Lattice_data.temporary_data.G_matrix_vec_y
			else
				error("Unknown direction: $direction")
			end
		end
		if length(G_out) < q_length
			G_out = [zeros(ComplexF64, 2*N, 2*N) for _ in eachindex(q_vecs)]
		end
		if length(q_proj) < q_length
			q_proj = zeros(ComplexF64, q_length)
			Lattice_data.temporary_data.G_q_proj = q_proj
		end
		q_proj_view = @view q_proj[1:q_length]
		FT_t = transpose(@view FT_vecs[:, 1:q_length])

		if direction == "x"
			for l in 1:twoN
				for m in 1:twoN
					build_G_r_from_S_r_x!(Lattice_data, G_r, l, m, 1)
					mul!(q_proj_view, FT_t, G_r)
					@inbounds for iq in 1:q_length
						G_out[iq][l, m] = q_proj_view[iq]
					end
				end
			end
		elseif direction == "y"
			for l in 1:twoN
				for m in 1:twoN
					build_G_r_from_S_r_y!(Lattice_data, G_r, l, m, 1)
					mul!(q_proj_view, FT_t, G_r)
					@inbounds for iq in 1:q_length
						G_out[iq][l, m] = q_proj_view[iq]
					end
				end
			end
		else
			error("Unknown direction: $direction")
		end
		return G_out, nothing
	end


	"""
	    make_S_matrix_q_vec(Lattice_data, q_vecs; spin=1, S_alloc=:S_up, direction="x")

	Older per-spin bond-current matrices S_σ(q)[l, m] = Σ_i e^{iq·r_i} S_r(i) (from
	`make_S_r`), one per q. Superseded by `make_G_matrix_q_vec`.
	"""
	function make_S_matrix_q_vec(Lattice_data::LatticeData, q_vecs::Array{Array{Float64,1},1}; spin::Int64=1, S_alloc=:S_up, direction::String="x")
		# Extract simulation parameters
		N = Lattice_data.sim_data.N
		X = Lattice_data.const_struct.x
		Y = Lattice_data.const_struct.y
		FT_vecs = [[exp(1im * (q[1]*X[i] + q[2]*Y[i])) for i in 1:N] for q in q_vecs] # precompute Fourier transform vectors


		# Choose preallocation based on direction
		if direction == "x"
			if S_alloc == :S_up
				S = Lattice_data.temporary_data.S_up_x
			elseif S_alloc == :S_down
				S = Lattice_data.temporary_data.S_down_x
			end
		elseif direction == "y"
			if S_alloc == :S_up
				S = Lattice_data.temporary_data.S_up_y
			elseif S_alloc == :S_down
				S = Lattice_data.temporary_data.S_down_y
			end
		else
			error("Unknown direction: $direction")
		end

		if length(q_vecs) > length(S)
			S = [zeros(ComplexF64, 2*N, 2*N) for _ in eachindex(q_vecs)] # pre-allocate S matrix vector
		end

		# Fill the S matrix
		for l in 1:2*N
			for m in 1:2*N
				lm_ix = CartesianIndex((l, m)) # Cartesian index for (l, m)
				S_r = make_S_r(Lattice_data, l, m, spin, 1; direction=direction) # real space S_r matrix
				# Apply Fourier transform to S_r
				for (i,q) in enumerate(q_vecs)
					S[i][lm_ix] = dot(FT_vecs[i], S_r)
				end
			end
		end
		return S
	end





	"""
	    current_correlation_K_4dir(data, w; δ=nothing, pre_Gx=nothing, pre_Gy=nothing) -> (Λxx, Λxy, Λyx, Λyy)

	q = 0 current-current correlation at real frequency ω + iδ, all four components:

	    Λ_ab(ω) = -(t²/(N·SC_N)) Σ_K Σ_lm G^a_lm conj(G^b_lm) (f_l - f_m)/(ω + E_l - E_m + iδ)

	δ defaults to max(E)/N; `pre_Gx`/`pre_Gy` reuse precomputed q = 0 vertices.
	Re Λ gives the Drude part and Im Λ/ω the regular conductivity. Reference for
	`measure_fixed_delta_fast!`.
	"""
	function current_correlation_K_4dir(data::LatticeData, w::Float64; δ=nothing, pre_Gx=nothing, pre_Gy=nothing)
		f =data.temporary_data.f
		Energies = data.current_state.Energies
		t = data.sim_data.t
		N = data.sim_data.N
		SC_N = data.sim_data.SC_L^2

		if !(typeof(δ) <: Number)
			δ = (maximum(data.current_state.Energies) / N)
		end

		Gx_all = pre_Gx === nothing ? make_G_matrix_K0(data; K_list=[1:SC_N;], direction="x")[1] : pre_Gx
		Gy_all = pre_Gy === nothing ? make_G_matrix_K0(data; K_list=[1:SC_N;], direction="y")[1] : pre_Gy

		Lxx = ComplexF64(0.0)
		Lxy = ComplexF64(0.0)
		Lyx = ComplexF64(0.0)
		Lyy = ComplexF64(0.0)

		@inbounds for K in 1:SC_N
			Gx = Gx_all[K]
			Gy = Gy_all[K]
			for l in 1:2N
				f_lK = f[l, K]
				E_lK = Energies[l, K]
				for m in 1:2N
					a = w + E_lK - Energies[m, K]
					a2 = (a^2 + δ^2)^(-1)
					A = a * a2
					lorenz = a2 * δ
					kernel = (f_lK - f[m, K]) * (A - 1im * lorenz)

					gx = Gx[l, m]
					gy = Gy[l, m]

					Lxx += gx * conj(gx) * kernel
					Lxy += gx * conj(gy) * kernel
					Lyx += gy * conj(gx) * kernel
					Lyy += gy * conj(gy) * kernel
				end
			end
		end

		pref = -t^2 / (N * SC_N)
		return Lxx * pref, Lxy * pref, Lyx * pref, Lyy * pref
	end

	"""
	    current_correlation_K_4dir_direct(data; pre_Gx=nothing, pre_Gy=nothing) -> (Λxx, Λxy, Λyx, Λyy)

	Direct ω → 0, δ → 0 limit of Re Λ_aa'(q = 0, ω + iδ) (the Drude-weight kernel), in closed
	form with no ω or δ ("Drude weight" note):

	    Re Λ_aa' = -(t²/N) Σ_{E_λ ≠ E_μ} (f_λ - f_μ)/(E_λ - E_μ) · Re[G^a_λμ conj(G^a'_λμ)]

	Degenerate pairs are skipped. Off-diagonal components use Re[G^a conj(G^a')], not |G|².
	Used for the `L_D_direct_*` measurement. Not valid with a magnetic field (it then gives
	the static value).
	"""
	function current_correlation_K_4dir_direct(data::LatticeData; pre_Gx=nothing, pre_Gy=nothing)
		f =data.temporary_data.f
		Energies = data.current_state.Energies
		t = data.sim_data.t
		N = data.sim_data.N
		SC_N = data.sim_data.SC_L^2

		Gx_all = pre_Gx === nothing ? make_G_matrix_K0(data; K_list=[1:SC_N;], direction="x")[1] : pre_Gx
		Gy_all = pre_Gy === nothing ? make_G_matrix_K0(data; K_list=[1:SC_N;], direction="y")[1] : pre_Gy

		Lxx = 0.0
		Lxy = 0.0
		Lyx = 0.0
		Lyy = 0.0

		@inbounds for K in 1:SC_N
			Gx = Gx_all[K]
			Gy = Gy_all[K]
			for l in 1:2N
				f_lK = f[l, K]
				E_lK = Energies[l, K]
				for m in 1:2N
					dE = E_lK - Energies[m, K]
					abs(dE) <= 1e-12 && continue  # skip degenerate pairs (ε_λ = ε_μ)
					kernel = (f_lK - f[m, K]) / dE

					gx = Gx[l, m]
					gy = Gy[l, m]

					Lxx += real(gx * conj(gx)) * kernel
					Lxy += real(gx * conj(gy)) * kernel
					Lyx += real(gy * conj(gx)) * kernel
					Lyy += real(gy * conj(gy)) * kernel
				end
			end
		end

		pref = -t^2 / (N * SC_N)
		return Lxx * pref, Lxy * pref, Lyx * pref, Lyy * pref
	end

	"""
	    current_correlation_K(data, w; δ=nothing, pre_G1=nothing, pre_G2=nothing) -> ComplexF64

	Λxx(q = 0, ω + iδ), one component of `current_correlation_K_4dir` (real part → Drude
	weight, imaginary part → regular conductivity).
	"""
	function current_correlation_K(data::LatticeData, w::Float64; δ=nothing, pre_G1=nothing, pre_G2=nothing)
		# Extract simulation parameters
		f = data.temporary_data.f
		Energies = data.current_state.Energies
		t = data.sim_data.t
		N = data.sim_data.N
		SC_N = data.sim_data.SC_L^2
		
		# δ is a small number for iw_n -> w + iδ
		if !(typeof(δ) <: Number)
			δ = (maximum(data.current_state.Energies) / N)  # this is a guess, can be changed later
		end

		G1_all = pre_G1 === nothing ? make_G_matrix_K0(data; K_list=[1:SC_N;], direction="x")[1] : pre_G1
		G2_all = pre_G2 === nothing ? make_G_matrix_K0(data; K_list=[1:SC_N;], direction="x")[1] : pre_G2

		Lxx = ComplexF64(0.0)
	
		# Compute the current-current correlation
		@inbounds for K in 1:SC_N
			G1 = G1_all[K]
			G2 = G2_all[K]

			for l in 1:2N
				f_lK = f[l, K]
				E_lK = Energies[l, K]
				for m in 1:2N
					a = w + E_lK - Energies[m, K]
					a2 = (a^2 + δ^2)^(-1)
					A = a * a2
					lorenz = a2 * δ
					Lxx += G1[l,m] * conj(G2[l,m]) * (f_lK - f[m, K]) * (A - 1im * lorenz)
				end
			end
		end
		Lxx *= - t^2 / (N * SC_N)

		return Lxx
	end

	


	"""
	    make_G_matrix_K0(Lattice_data; K_list=[1], direction="x") -> (G_out, nothing)

	q = 0 current vertices G^a for each supercell sector in `K_list` (the sum over sites of
	`build_G_r_from_S_r!`), written into the `G_matrix_vec_x/_y` buffers.
	"""
	function make_G_matrix_K0(Lattice_data::LatticeData; K_list=[1], direction::String="x")
		N = Lattice_data.sim_data.N
		twoN = 2 * N
		ensure_G_buffers!(Lattice_data) # G_matrix_vec_x/_y are allocated on first use
		G_r = Lattice_data.temporary_data.G_r_vec
		# Choose preallocation for G_out
		if direction == "x"
			G_out = Lattice_data.temporary_data.G_matrix_vec_x
		elseif direction == "y"
			G_out = Lattice_data.temporary_data.G_matrix_vec_y
		else
			error("Unknown direction: $direction")
		end
		if length(K_list) > length(G_out)
			G_out = [zeros(ComplexF64, 2*N, 2*N) for _ in eachindex(K_list)] # pre-allocate G matrix vector	
		end

		# Fill the G matrix directly from real-space combined G_r at q=0.
		for (i, K) in enumerate(K_list)
			for l in 1:twoN
				for m in 1:twoN
					build_G_r_from_S_r!(Lattice_data, G_r, l, m, K; direction=direction)
					@inbounds G_out[i][l, m] = sum(G_r)
				end
			end
		end
		return G_out, nothing
	end


	"""
	    get_K_plus_Q_idx(K_idx, Q_idx, L_s) -> Int

	Index of the supercell wavevector K + Q (periodic in the L_s × L_s supercell grid).
	"""
	function get_K_plus_Q_idx(K_idx::Int64, Q_idx::Int64, L_s::Int64)
		Qx, Qy = get_x(Q_idx, L_s), get_y(Q_idx, L_s)
		Kx, Ky = get_x(K_idx, L_s), get_y(K_idx, L_s)
		KpQx, KpQy = add_xy_ix(Kx, Qx, L_s), add_xy_ix(Ky, Qy, L_s)
		KpQ_idx = idx(KpQx, KpQy, L_s)
		return KpQ_idx
	end


	"""
	    make_S_matrix_K0(Lattice_data; K_list=[1], spin=1, S_alloc=:S_up, direction="x")

	q = 0 per-spin bond-current matrices S_σ[l, m] = Σ_i S_r(i) (from `make_S_r`) for each
	sector in `K_list` (spin -1 uses the -K sectors). Older form of `make_G_matrix_K0`.
	"""
	function make_S_matrix_K0(Lattice_data::LatticeData; K_list=[1], spin=1, S_alloc=:S_up, direction::String="x")
		# Extract simulation parameters
		N =Lattice_data.sim_data.N
		# maby beter outside of this function
		if spin == -1
			K_list = Lattice_data.const_struct.SC_minus_K[K_list]
		end

		# Choose preallocation based on direction
		if direction == "x"
			if S_alloc == :S_up
				S = Lattice_data.temporary_data.S_up_x
			elseif S_alloc == :S_down
				S = Lattice_data.temporary_data.S_down_x
			end
		elseif direction == "y"
			if S_alloc == :S_up
				S = Lattice_data.temporary_data.S_up_y
			elseif S_alloc == :S_down
				S = Lattice_data.temporary_data.S_down_y
			end
		else
			error("Unknown direction: $direction")
		end

		if length(K_list) > length(S)
			S = [zeros(ComplexF64, 2*N, 2*N) for _ in eachindex(K_list)] # pre-allocate S matrix vector
		end

		# Fill the S matrix
		for (i, K) in enumerate(K_list)
			for l in 1:2*N
				for m in 1:2*N
					lm_ix = CartesianIndex((l, m)) # Cartesian index for (l, m)
					S_r = make_S_r(Lattice_data, l, m, spin, K, direction=direction)
					# A FT for q=0 is just the sum over all r
					S[i][lm_ix] = sum(S_r)
				end
			end
		end

		return S
	end
	
	"""
	    _fill_S_r_spin!(S_r, U_l, U_m, new_ix_list, linear_index_list, linear_length, phase_Tr,
	                    peierls_phases, N, lat_add, Val(spin)) -> S_r

	Kernel of `make_S_r`: the bond current of levels (l, m) on every bond i → i + a,
	S_r(i) = conj(T_i) conj(U_l[i+a]) U_m[i] - T_i conj(U_l[i]) U_m[i+a] for spin up
	(complex-conjugated factors for spin down), T_i = boundary twist × Peierls phase.
	One method per spin, so there is no spin branch inside the loop.
	"""
	@inline function _fill_S_r_spin!(S_r::Array{ComplexF64,1},
									 U_l::AbstractVector{ComplexF64},
									 U_m::AbstractVector{ComplexF64},
									 new_ix_list::Array{Int64,1},
									 linear_index_list::Array{Int64,1},
									 linear_length::Int64,
									 phase_Tr::ComplexF64,
									 peierls_phases,
									 N::Int64,
									 lat_add::Int64,
									 ::Val{1})
		@inbounds for ix in 1:N
			new_ix = new_ix_list[ix]
			lat_ix = ix + lat_add
			new_lat_ix = new_ix + lat_add
			Tr = linear_index_list[ix] == linear_length ? phase_Tr : (1.0 + 0.0im)
			Tr *= peierls_phases[ix]

			U1_term = conj(U_l[new_lat_ix]) * U_m[lat_ix]
			U2_term = conj(U_l[lat_ix]) * U_m[new_lat_ix]

			S_r[ix] = conj(Tr) * U1_term - Tr * U2_term
		end
		return S_r
	end

	@inline function _fill_S_r_spin!(S_r::Array{ComplexF64,1},
									 U_l::AbstractVector{ComplexF64},
									 U_m::AbstractVector{ComplexF64},
									 new_ix_list::Array{Int64,1},
									 linear_index_list::Array{Int64,1},
									 linear_length::Int64,
									 phase_Tr::ComplexF64,
									 peierls_phases,
									 N::Int64,
									 lat_add::Int64,
									 ::Val{-1})
		@inbounds for ix in 1:N
			new_ix = new_ix_list[ix]
			lat_ix = ix + lat_add
			new_lat_ix = new_ix + lat_add
			Tr = linear_index_list[ix] == linear_length ? phase_Tr : (1.0 + 0.0im)
			Tr *= peierls_phases[ix]

			# Algebraically identical to the previous spin==-1 conjugation branch.
			U1_term = U_l[new_lat_ix] * conj(U_m[lat_ix])
			U2_term = U_l[lat_ix] * conj(U_m[new_lat_ix])

			S_r[ix] = conj(Tr) * U1_term - Tr * U2_term
		end
		return S_r
	end

	"""
	    _make_S_r_direction!(Lattice_data, l, m, spin, k, new_ix_list, linear_index_list,
	                         linear_length, phase_Tr, peierls_phases) -> S_r

	Call `_fill_S_r_spin!` for one spin, writing into `temporary_data.S_r_vec`.
	"""
	@inline function _make_S_r_direction!(Lattice_data::LatticeData, l::Int64, m::Int64, spin::Int64, k::Int64,
									 new_ix_list::Array{Int64,1},
									 linear_index_list::Array{Int64,1},
									 linear_length::Int64,
									 phase_Tr::ComplexF64,
									 peierls_phases)
		N = Lattice_data.sim_data.N
		U_matrix = Lattice_data.current_state.U_matrix
		S_r = Lattice_data.temporary_data.S_r_vec
		U_l = @view U_matrix[:, l, k]
		U_m = @view U_matrix[:, m, k]

		if spin == 1
			return _fill_S_r_spin!(S_r, U_l, U_m, new_ix_list, linear_index_list, linear_length, phase_Tr, peierls_phases, N, 0, Val(1))
		elseif spin == -1
			return _fill_S_r_spin!(S_r, U_l, U_m, new_ix_list, linear_index_list, linear_length, phase_Tr, peierls_phases, N, N, Val(-1))
		else
			error("spin must be ±1")
		end
	end



	"""
	    make_S_r(Lattice_data, l, m, spin, k; direction="x") -> Vector{ComplexF64}

	Real-space bond current of the level pair (l, m), spin `spin` (±1), sector `k`, on every
	bond along `direction` (see `_fill_S_r_spin!`). Returns the shared scratch vector.
	"""
	function make_S_r(Lattice_data::LatticeData, l::Int64, m::Int64, spin::Int64, k::Int64; direction::String="x")
		Lx = Lattice_data.sim_data.Lx
		Ly = Lattice_data.sim_data.Ly
		if direction == "y"
			phase_Tr = exp(-1im * 2π/Ly * Lattice_data.const_struct.SC_K_vectors[k][2]) # phase factor for translation in y direction
			new_ix_list = Lattice_data.const_struct.new_iy_list
			linear_index_list = Lattice_data.const_struct.y
			linear_length = Ly
			peierls_phases = view(Lattice_data.const_struct.peierls_phases, :, 2)
			return _make_S_r_direction!(Lattice_data, l, m, spin, k,
				new_ix_list, linear_index_list, linear_length, phase_Tr, peierls_phases)
		elseif direction == "x"
			phase_Tr = exp(-1im * 2π/Lx * Lattice_data.const_struct.SC_K_vectors[k][1]) # phase factor for translation in x direction
			new_ix_list = Lattice_data.const_struct.new_ix_list
			linear_index_list = Lattice_data.const_struct.x
			linear_length = Lx
			peierls_phases = view(Lattice_data.const_struct.peierls_phases, :, 1)
			return _make_S_r_direction!(Lattice_data, l, m, spin, k,
				new_ix_list, linear_index_list, linear_length, phase_Tr, peierls_phases)
		else
			error("direction must be 'x' or 'y'")
		end
	end


	"""
	    measure_conductivities_fixed_delta!(data, w, i, j, d_idx; direction1="x", direction2="x",
	                                        pre_G1=nothing, pre_G2=nothing, δ=0.1)

	Store one component Λ_ab(ω + iδ) (`current_correlation_K`) at measurement `i`, broadening
	index `d_idx`, frequency index `j`. Not used by the pipeline (the measurement fills all
	four components at once).
	"""
	function measure_conductivities_fixed_delta!(data::LatticeData, w::Float64, i::Int, j::Int, d_idx::Int; direction1::String="x", direction2::String="x", pre_G1=nothing, pre_G2=nothing, δ::Float64=0.1)
		if pre_G1 === nothing
			G1 = make_G_matrix_K0(data; K_list=[1:data.sim_data.SC_L^2;], direction=direction1)[1]
		else
			G1 = pre_G1
		end
		if pre_G2 === nothing
			G2 = make_G_matrix_K0(data; K_list=[1:data.sim_data.SC_L^2;], direction=direction2)[1]
		else
			G2 = pre_G2
		end
		measurements = data.measurement_data
		
		if (direction1, direction2) == ("x","x")
			L_w = measurements.Lxx_w_fixed
		elseif (direction1, direction2) == ("y","y")
			L_w = measurements.Lyy_w_fixed
		elseif (direction1, direction2) == ("x","y")
			L_w = measurements.Lxy_w_fixed
		elseif (direction1, direction2) == ("y","x")
			L_w = measurements.Lyx_w_fixed
		else
			error("Invalid direction combination: ($direction1, $direction2)")
		end

		# Compute and store with extra delta dimension: [i][d_idx][j]
		L_w[i][d_idx][j] = current_correlation_K(data, w; pre_G1=G1, pre_G2=G2, δ=δ)
	end

	##
	# DC conductivity and Drude weight (delta->0 integrated formulas)

	"""
		integral_kernel_PV(dE::Float64, dw::Float64) -> Float64

	Compute the principal value integral:
	  p.v. ∫_{-Δω}^{Δω} dω / [ω (ΔE + ω)] = (1/ΔE) ln|( Δω - ΔE )/( Δω + ΔE )|
	Special case: |ΔE| ≈ 0 → limit is -2/Δω (from L'Hôpital).
	Special case: |ΔE| ≈ Δω → diverges logarithmically; clamped to a large finite value.
	"""
	function integral_kernel_PV(dE::Float64, dw::Float64)
		abs_dE = abs(dE)
		if abs_dE < 1e-14
			# lim_{ΔE→0} (1/ΔE) ln|(Δω-ΔE)/(Δω+ΔE)| = -2/Δω
			return -2.0 / dw
		end
		ratio = abs((dw - dE) / (dw + dE))
		if ratio < 1e-30
			# |ΔE| ≈ Δω: logarithm diverges, clamp
			return  -30* log(10) / dE
		end
		return log(ratio) / dE
	end

	"""
		integral_kernel_log(dE::Float64, dw::Float64) -> Float64

	Compute the integral:
	  (1/Δω) ∫_0^{Δω} dω / (ΔE + ω) = (1/Δω) ln((ΔE + Δω) / |ΔE|)
	where ΔE = ε_λ - ε_μ. For |ΔE| < Δω the pole is inside [0,Δω] and the
	P.V. still gives this formula.
	Special case: |ΔE| ≈ 0 → diverges; returns a large clamped value.
	"""
	function integral_kernel_log(dE::Float64, dw::Float64)
		abs_dE = abs(dE)
		if abs_dE < 1e-14
			# diverges like ln(Δω/|ΔE|)/Δω, clamp
			return log(dw / 1e-14) / dw
		end
		arg = abs((dE + dw) / dE)
		if arg < 1e-30
			return -30.0 * log(10.0) / dw
		end
		return log(arg) / dw
	end

	"""
		conductivity_dc(data::LatticeData, dw::Float64; direction1="x", direction2="x", pre_G1=nothing, pre_G2=nothing)

	Compute the DC regular conductivity using the delta->0 integrated formula,
	averaged over the symmetric window [-Δω, Δω]:

	  σ_μν^dc(Δω) = -(t²/N) (1/2Δω) Σ_{λ,m} (f_λ - f_m) ×
	    [ Im[G^a_λm (G^a'_λm)*] · I₁(Δε, Δω)  +  (π/Δε) Re[G^a_λm (G^a'_λm)*] · Θ(Δω - |Δε|) ]

	where I₁ = p.v. ∫_{-Δω}^{Δω} 1/(ω(Δε+ω)) dω  (integral_kernel_PV).
	For diagonal μ=ν, Im[|G|²]=0, and this reduces to the Heaviside-only formula.
	Only K=1 (no supercell sum) for now.
	"""
	function conductivity_dc(data::LatticeData, dw::Float64; direction1::String="x", direction2::String="x", pre_G1=nothing, pre_G2=nothing)
		# Extract simulation parameters
		f = data.temporary_data.f
		Energies = data.current_state.Energies
		t = data.sim_data.t
		N = data.sim_data.N
		is_diagonal = (direction1 == direction2)
		
		# Get G matrices (K=1 only)
		if pre_G1 === nothing
			G1 = make_G_matrix_K0(data; K_list=[1], direction=direction1)[1][1]
		else
			G1 = pre_G1[1]
		end
		if pre_G2 === nothing
			G2 = make_G_matrix_K0(data; K_list=[1], direction=direction2)[1][1]
		else
			G2 = pre_G2[1]
		end
		
		sigma_dc = ComplexF64(0.0)
		
		# Sum over eigenstate pairs
		for l in 1:2N
			for m in 1:2N
				dE = Energies[l] - Energies[m]  # Δε = ε_λ - ε_μ
				df = f[l] - f[m]
				abs_dE = abs(dE)
				GG_star = G1[l,m] * conj(G2[l,m])
				
				# Term B: (π/Δε) Re[GG*] · Θ(Δω - |Δε|) — delta-function part
				# Skip Δε ≈ 0 (f_l - f_m = 0 then, so no contribution)
				if abs_dE > 1e-14 && abs_dE < dw
					sigma_dc += df * (pi / dE) * real(GG_star)
				end
				
				# Term A: Im[GG*] · I₁(Δε, Δω) — P.V. part (only for off-diagonal)
				if !is_diagonal
					sigma_dc += df * imag(GG_star) * integral_kernel_PV(dE, dw)
				end
			end
		end
		
		sigma_dc *= -t^2 / (2 * dw * N)
		
		# this is real for both diagonal and off-diagonal, but take real part to be safe
		return real(sigma_dc)

	end

	"""
		drude_weight_dc(data::LatticeData, dw::Float64; direction1="x", direction2="x", pre_G1=nothing, pre_G2=nothing)

	Compute the Drude weight kernel using the delta->0 integrated formula,
	averaged over [0, Δω]:

	  Λ_D^{μν}(Δω) = -(t²/N) Σ_{λ,m} (f_λ - f_m) ×
	    [ Re[GG*] · I₂(Δε, Δω)  +  (π/Δω) Im[GG*] · Θ(Δω - |Δε|) ]

	where I₂ = (1/Δω) ln((Δε + Δω) / |Δε|)  (integral_kernel_log).
	Only K=1 (no supercell sum) for now.
	"""
	function drude_weight_dc(data::LatticeData, dw::Float64; direction1::String="x", direction2::String="x", pre_G1=nothing, pre_G2=nothing)
		# Extract simulation parameters
		f = data.temporary_data.f
		Energies = data.current_state.Energies
		t = data.sim_data.t
		N = data.sim_data.N
		is_diagonal = (direction1 == direction2)
		
		# Get G matrices (K=1 only)
		if pre_G1 === nothing
			G1 = make_G_matrix_K0(data; K_list=[1], direction=direction1)[1][1]
		else
			G1 = pre_G1[1]
		end
		if pre_G2 === nothing
			G2 = make_G_matrix_K0(data; K_list=[1], direction=direction2)[1][1]
		else
			G2 = pre_G2[1]
		end
		
		Lambda_D = ComplexF64(0.0)
		
		# Sum over eigenstate pairs
		for l in 1:2N
			for m in 1:2N
				dE = Energies[l] - Energies[m]  # Δε = ε_λ - ε_μ
				df = f[l] - f[m]
				abs_dE = abs(dE)
				GG_star = G1[l,m] * conj(G2[l,m])
				
				# P.V. term: Re[GG*] · I₂(Δε, Δω)
				# Skip Δε ≈ 0 (divergent kernel, but f_l - f_m = 0)
				if abs_dE > 1e-14
					Lambda_D += df * real(GG_star) * integral_kernel_log(dE, dw)
				end
				
				# δ-function term: (π/Δω) Im[GG*] · Θ(Δω - |Δε|) (only for off-diagonal)
				if !is_diagonal && abs_dE > 1e-14 && abs_dE < dw
					Lambda_D += df * imag(GG_star) * pi / dw
				end
			end
		end
		
		Lambda_D *= -t^2 / N
		
		# this is real for both diagonal and off-diagonal, but take real part to be safe
		return real(Lambda_D)

	end

	"""
		measure_conductivity_dc!(data::LatticeData, i::Int; pre_Gx=nothing, pre_Gy=nothing)

	Measure DC conductivity for all direction pairs and all Δω windows.
	Stores results in measurements.conductivity_dc_{xx,xy,yx,yy}[i][dw_idx].
	"""
	function measure_conductivity_dc!(data::LatticeData, i::Int; pre_Gx=nothing, pre_Gy=nothing)
		measurements = data.measurement_data
		
		for (dw_idx, dw) in enumerate(data.sim_data.dw_values)
			measurements.conductivity_dc_xx[i][dw_idx] = conductivity_dc(data, dw; direction1="x", direction2="x", pre_G1=pre_Gx, pre_G2=pre_Gx)
			measurements.conductivity_dc_xy[i][dw_idx] = conductivity_dc(data, dw; direction1="x", direction2="y", pre_G1=pre_Gx, pre_G2=pre_Gy)
			measurements.conductivity_dc_yx[i][dw_idx] = conductivity_dc(data, dw; direction1="y", direction2="x", pre_G1=pre_Gy, pre_G2=pre_Gx)
			measurements.conductivity_dc_yy[i][dw_idx] = conductivity_dc(data, dw; direction1="y", direction2="y", pre_G1=pre_Gy, pre_G2=pre_Gy)
		end
		return nothing
	end

	"""
		measure_L_D_dc!(data::LatticeData, i::Int; pre_Gx=nothing, pre_Gy=nothing)

	Measure L_D (DC, integrated current-current correlator) for all direction pairs and all Δω windows.
	Stores results in measurements.L_D_dc_{xx,xy,yx,yy}[i][dw_idx].
	"""
	function measure_L_D_dc!(data::LatticeData, i::Int; pre_Gx=nothing, pre_Gy=nothing)
		measurements = data.measurement_data
		
		for (dw_idx, dw) in enumerate(data.sim_data.dw_values)
			measurements.L_D_dc_xx[i][dw_idx] = drude_weight_dc(data, dw; direction1="x", direction2="x", pre_G1=pre_Gx, pre_G2=pre_Gx)
			measurements.L_D_dc_xy[i][dw_idx] = drude_weight_dc(data, dw; direction1="x", direction2="y", pre_G1=pre_Gx, pre_G2=pre_Gy)
			measurements.L_D_dc_yx[i][dw_idx] = drude_weight_dc(data, dw; direction1="y", direction2="x", pre_G1=pre_Gy, pre_G2=pre_Gx)
			measurements.L_D_dc_yy[i][dw_idx] = drude_weight_dc(data, dw; direction1="y", direction2="y", pre_G1=pre_Gy, pre_G2=pre_Gy)
		end
		return nothing
	end

	"""
		measure_L_D_direct!(data::LatticeData, i::Int; pre_Gx=nothing, pre_Gy=nothing)

	Measure L_D_direct = Re[Λ_aa'(q=0, ω→0, δ→0)] via the closed-form direct limit
	(no ω, no δ). Stores the four tensor components in
	measurements.L_D_direct_{xx,xy,yx,yy}[i].
	"""
	function measure_L_D_direct!(data::LatticeData, i::Int; pre_Gx=nothing, pre_Gy=nothing)
		measurements = data.measurement_data

		Lxx, Lxy, Lyx, Lyy = current_correlation_K_4dir_direct(data; pre_Gx=pre_Gx, pre_Gy=pre_Gy)
		measurements.L_D_direct_xx[i] = Lxx
		measurements.L_D_direct_xy[i] = Lxy
		measurements.L_D_direct_yx[i] = Lyx
		measurements.L_D_direct_yy[i] = Lyy
		return nothing
	end

	##
	# Susceptibility observable (spin/charge)
	"""
	    susceptibility_q_vec_both!(Lattice_data, q_vecs, chi_spin, chi_charge) -> (chi_spin, chi_charge)

	Static (ω = 0) spin and charge susceptibilities χ(q) of the current configuration for
	each q in `q_vecs`, written into `chi_spin` and `chi_charge` (K = 0 sector).
	With the density vertices K_σ(q) in the eigenbasis and M_lm as in the current
	correlation, χ_spin = -½ Σ_lm |K_↑ + K_↓ᵀ|² M_lm and
	χ_charge = ½ (4β |Σ_l K_↑,ll f_l|² - Σ_lm |K_↑ - K_↓ᵀ|² M_lm).
	Uses the current-vertex buffers as scratch. The measurement uses q = 0 only.
	"""
	function susceptibility_q_vec_both!(Lattice_data::LatticeData,
									q_vecs::Array{Array{Float64,1},1},
									chi_spin::AbstractVector{Float64},
									chi_charge::AbstractVector{Float64})
		# Extract simulation parameters
		Energies = Lattice_data.current_state.Energies
		f = Lattice_data.temporary_data.f
		beta = Lattice_data.sim_data.beta
		N = Lattice_data.sim_data.N
		twoN = 2 * N
		U_matrix = Lattice_data.current_state.U_matrix
		U_K1 = @view U_matrix[:, :, 1]
		U_up = @view U_K1[1:N, :]
		U_dn = @view U_K1[(N+1):(2*N), :]
		Energies_K1 = @view Energies[:, 1]
		f_K1 = @view f[:, 1]
		q_length = length(q_vecs)
		FT_vecs = get_FT_table_for_q_vecs!(Lattice_data, q_vecs)

		length(chi_spin) >= q_length || error("chi_spin output too short")
		length(chi_charge) >= q_length || error("chi_charge output too short")

		ensure_G_buffers!(Lattice_data) # used as scratch here; allocated on first use
		K_up_qs = Lattice_data.temporary_data.G_matrix_vec_x
		K_dn_qs = Lattice_data.temporary_data.G_matrix_vec_y
		if length(K_up_qs) < q_length
			K_up_qs = [zeros(ComplexF64, twoN, twoN) for _ in 1:q_length]
			Lattice_data.temporary_data.G_matrix_vec_x = K_up_qs
		end
		if length(K_dn_qs) < q_length
			K_dn_qs = [zeros(ComplexF64, twoN, twoN) for _ in 1:q_length]
			Lattice_data.temporary_data.G_matrix_vec_y = K_dn_qs
		end

		W_up = Lattice_data.temporary_data.sus_weighted_up
		W_dn_conj = Lattice_data.temporary_data.sus_weighted_dn_conj
		if size(W_up, 1) != N || size(W_up, 2) != twoN
			W_up = zeros(ComplexF64, N, twoN)
			Lattice_data.temporary_data.sus_weighted_up = W_up
		end
		if size(W_dn_conj, 1) != N || size(W_dn_conj, 2) != twoN
			W_dn_conj = zeros(ComplexF64, N, twoN)
			Lattice_data.temporary_data.sus_weighted_dn_conj = W_dn_conj
		end

		@inbounds for iq in 1:q_length
			K_up = K_up_qs[iq]
			K_dn = K_dn_qs[iq]

			# Build weighted row blocks for this q and form K_up/K_dn via BLAS matmul.
			for r in 1:N
				phase = FT_vecs[r, iq]
				for c in 1:twoN
					W_up[r, c] = phase * U_up[r, c]
					W_dn_conj[r, c] = phase * conj(U_dn[r, c])
				end
			end
			mul!(K_up, adjoint(U_up), W_up)
			mul!(K_dn, transpose(U_dn), W_dn_conj)

			# Drude-like term appears only in charge susceptibility.
			S_charge = ComplexF64(0.0)
			for l in 1:twoN
				S_charge += K_up[l,l] * f_K1[l]
			end
			drude_charge = 4.0 * beta * abs2(S_charge)

			sum_spin = 0.0
			sum_charge = 0.0
			for l in 1:twoN
				f_l = f_K1[l]
				for m in 1:twoN
					diff = Energies_K1[l] - Energies_K1[m]
					if abs(diff) < 1e-12
						Mlm = -beta * f_l * (1 - f_l)
					else
						Mlm = (f_l - f_K1[m]) / diff
					end

					Kdn_T = K_dn[m, l]
					Kup = K_up[l, m]
					L_charge = Kup - Kdn_T
					L_spin = Kup + Kdn_T

					sum_charge += abs2(L_charge) * Mlm
					sum_spin += abs2(L_spin) * Mlm
				end
			end

			chi_charge[iq] = 0.5 * (drude_charge - sum_charge)
			chi_spin[iq] = -0.5 * sum_spin
		end

		return chi_spin, chi_charge
	end

	"""
	    susceptibility_q_vec(Lattice_data, q_vecs; type=:spin) -> Vector{Float64}

	The spin (`type=:spin`) or charge (`:charge`) susceptibility χ(q) from
	`susceptibility_q_vec_both!`.
	"""
	function susceptibility_q_vec(Lattice_data::LatticeData, q_vecs::Array{Array{Float64,1},1}; type::Symbol=:spin)
		if type == :spin
			chi_spin = zeros(Float64, length(q_vecs))
			chi_charge = zeros(Float64, length(q_vecs))
			susceptibility_q_vec_both!(Lattice_data, q_vecs, chi_spin, chi_charge)
			return chi_spin
		elseif type == :charge
			chi_spin = zeros(Float64, length(q_vecs))
			chi_charge = zeros(Float64, length(q_vecs))
			susceptibility_q_vec_both!(Lattice_data, q_vecs, chi_spin, chi_charge)
			return chi_charge
		else
			error("type must be :spin or :charge")
		end
	end


	"""
	    compute_DOS(data, w; δ=nothing) -> Float64

	(1/N) Σ over all 2N BdG levels of a Lorentzian of width δ at ω. Not used by the pipeline
	(the measured DOS is the sum of the LDOS).
	"""
	function compute_DOS(data::LatticeData, w; δ=nothing)
		N = data.sim_data.N
		SC_N = data.sim_data.SC_L^2
		Energies = data.current_state.Energies

		if δ === nothing
			δ = maximum([data.const_struct.w_delta, maximum(Energies)/(N*SC_N)])
		end

		DOS = 0.0
		for l in 1:2*N
			for k in 1:SC_N
				lorenz = (δ/pi) * (δ^2 +  (Energies[l, k] - w)^2)^-1
				DOS += lorenz
			end
		end
		DOS *= N^-1

		return DOS
	end


	"""
	    compute_Ps_new(data) -> Float64

	Another O(N²) formula for the pairing structure factor Ps (its old comment says it gives
	the same results). Not used by the pipeline.
	"""
	function compute_Ps_new(data::LatticeData)
		N = data.sim_data.N
		Ps = Float64(0.0)
		Ps_add = Float64(0.0)
		for i in 1:N
			Ps_add += sum(g_vec(data, i, 1, i, -1; g=:g_A))
			for j in 1:N
				gij_sum = sum(g_vec(data,i,1,j,1; g=:g_B))
				if i==j
					Ps += gij_sum * (gij_sum - 1)
				else
					Ps += gij_sum * gij_sum
				end
			end
		end
		Ps += abs2(Ps_add) 
		Ps *= 2 / N
		Ps += 1
		return real(Ps)
	end


	"""
	    compute_LDOS(data, ix, w; delt=nothing)

	Old single-site LDOS. Broken (reads `data.U_matrix` and `data.Energies`, which are not
	fields of `LatticeData`) and not used; see `compute_full_LDOS`.
	"""
	function compute_LDOS(data::LatticeData, ix::Int64, w::Float64; delt=nothing)
		N = data.sim_data.N
		U_matrix = data.U_matrix
		Energies = data.Energies
		if delt === nothing
			delt = minimum([maximum(Energies) / N, data.const_struct.w_delta])
		end

		LDOS = Float64(0.0)
		for l in 1:(2*N)
			LDOS += 2* abs2(U_matrix[ix, l]) * (delt/pi) * (delt^2 + (w - Energies[l])^2)^-1
		end
		return LDOS
	end

	"""
	    compute_full_LDOS(data, w; delt=nothing) -> Vector{Float64}

	Local density of states of every site at ω:
	LDOS_i(ω) = (2/SC_N) Σ_K Σ_l |U[i, l, K]|² L_δ(ω - E_l,K) (factor 2 for spin), with a
	Lorentzian of width `delt` (default max(max E/(N·SC_N), w_delta)). Written into
	`temporary_data.LDOS_vec`. Reference for the LDOS part of `spectral_fast!`.
	"""
	function compute_full_LDOS(data::LatticeData, w::Float64; delt=nothing)
		N = data.sim_data.N
		mu = data.sim_data.mu
		SC_N = data.sim_data.SC_L^2
		U_matrix = data.current_state.U_matrix
		Energies = data.current_state.Energies
		if delt === nothing
			delt = maximum([maximum(Energies) / (N*SC_N), data.const_struct.w_delta])
		end


		LDOS = fill!(data.temporary_data.LDOS_vec, 0.0)
		for l in 1:(2*N)
			for k in 1:SC_N
				@inbounds lorentz = (delt/pi) * (delt^2 + (w - (Energies[l, k] ))^2)^-1
				for ix in 1:N
					@inbounds LDOS[ix] += abs2(U_matrix[ix, l, k]) *  lorentz
				end
			end
		end
		LDOS .*= 2 / SC_N
		return LDOS
	end


	"""
	    find_Delta_SG(LDOS, w_list; threshold=0.01) -> Float64

	Spectral gap of one LDOS curve: the ω > 0 of its largest value above `threshold`
	(`w_list` must contain 0.0). Not used by the pipeline; see `get_SG` in sim_num_control.jl.
	"""
	function find_Delta_SG(LDOS::Array{Float64,1}, w_list::Array{Float64,1}; threshold=0.01)
		zero_ix = findfirst(==(0.0), w_list)
		if zero_ix === nothing
			throw(ArgumentError("w_list must contain 0.0"))
			return nothing
		end

		# Find the w that gives the maximum of LDOS for w > 0 and above threshold
		max_LDOS = 0.0
		max_ix = 1
		for i in (zero_ix+1):length(w_list)
			if LDOS[i] > threshold && LDOS[i] > max_LDOS
				max_LDOS = LDOS[i]
				max_ix = i
			end
		end

		return w_list[max_ix]
	end


	"""
	    compute_phase_correlation(data, r_vec; pairing_field=:Delta_s) -> Float64

	Phase correlation of the pairing field at separation `r_vec`:
	(1/N) Σ_j cos(θ_j - θ_{j+r}), θ = arg Δ (amplitudes ignored).
	"""
	function compute_phase_correlation(data::LatticeData, r_vec::Array{Int64,1}; pairing_field=:Delta_s)
		Delta =getfield(data.current_state, pairing_field)
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		N = data.sim_data.N
		rx = r_vec[1]
		ry = r_vec[2]

		# Compute the phase correlation using the delta operator and the provided r_vec
		phase_correlation = Float64(0.0)
		for j in 1:N
			j_r = idx(add_xy_ix(data.const_struct.x[j], rx, Lx), add_xy_ix(data.const_struct.y[j], ry, Ly), Lx)
			dj = Delta[j]
			djr = Delta[j_r]
			phase_correlation += real(exp(1im * angle(dj)) * exp(-1im * angle(djr)))
		end
		phase_correlation  /= N
		return phase_correlation
	end



	"""
	    compute_mixed_phase_correlation(data, l; pairing_field1=:Delta_dx, pairing_field2=:Delta_dy) -> Float64

	Cross phase correlation of two bond fields: (1/N) Σ_j cos(θ1_j - θ2_{j+l}) (d-wave, experimental).
	"""
	function compute_mixed_phase_correlation(data::LatticeData, l::Array{Int64,1}; pairing_field1=:Delta_dx, pairing_field2=:Delta_dy)
		Delta1 =getfield(data.current_state, pairing_field1)
		Delta2 = getfield(data.current_state, pairing_field2)
		Lx = data.sim_data.Lx
		Ly = data.sim_data.Ly
		N = data.sim_data.N
		lx = l[1]
		ly = l[2]

		# Compute the d-wave phase correlation using the delta operator and the provided l vector
		phase_correlation = Float64(0.0)
		for j in 1:N
			j_l = idx(add_xy_ix(data.const_struct.x[j], lx, Lx), add_xy_ix(data.const_struct.y[j], ly, Ly), Lx)
			dj1 = Delta1[j]
			djl2 = Delta2[j_l]
			phase_correlation += real(exp(1im * angle(dj1)) * exp(-1im * angle(djl2)))
		end
		phase_correlation  /= N
		return phase_correlation
	end

	"""
	    compute_U4(data)

	Old Binder-cumulant attempt 1 - |M|⁴/(2|M|²)² of ONE configuration, M = Σ_i Δ_i/N.
	Broken (reads `current_state.Delta`, which does not exist) and not used: the measured
	"U4" stream holds the K0 action instead.
	"""
	function compute_U4(data::LatticeData)
		numi =0.0
		denumi = 0.0
		U4 = 0.0
		Delta = data.current_state.Delta
		N = data.sim_data.N
		for i in 1:N
			di = Delta[i]
			for j in 1:N
				dj = Delta[j]
				denumi += di * conj(dj)
				for k in 1:N
					dk = Delta[k]
					for l in 1:N
						dl = Delta[l]
						numi += di * conj(dj) * dk * conj(dl)
					end
				end
			end
		end
		U4 = 1.0 - 0.5* (numi/(denumi)^2)
		return real(U4)
	end

	"""
	    compute_Sq(data, q::Tuple) -> ComplexF64

	Pairing-field structure factor of one configuration,
	S(q) = (1/N²) Σ_ij Δ_i conj(Δ_j) e^{iq·(r_j - r_i)}. Not used by the pipeline
	(`S_Delta_Q0` in `get_means` is the q = 0 value from the `Delta_mean` stream).
	"""
	function compute_Sq(data::LatticeData, q::Tuple{Float64,Float64})
    N = data.sim_data.N
    X = data.const_struct.x
    Y = data.const_struct.y
    Delta = data.current_state.Delta_s
    S_q = 0.0
    for ix1 in 1:N
        x1 = X[ix1]
        y1 = Y[ix1]
        r1 = [x1, y1]
        D1 = Delta[ix1]
        for ix2 in 1:N
            x2 = X[ix2]
            y2 = Y[ix2]
            r2 = [x2, y2]
            delta_r = r2 - r1
            D2 = conj(Delta[ix2])
            phase = exp(im * (q[1]*delta_r[1] + q[2]*delta_r[2]))
            S_q += D1 * D2 * phase
        end
    end
    return S_q / N^2
end

end

begin # U_puddles
	"""
	    look_U_puddles(L; U_disorder="gaussian", U=4.0, VI=4.0, λ=0.7, V0=0.0, impurity=0.075,
	                   rz=0.7, n=1000, seed=rand(Int64), signs=[1])

	Quick look at a U-disorder model: heatmap of one L×L map (`make_U_params`), titled with
	the mean, max and min of U over `n` random maps (`U_puddles_stat`).
	"""
	function look_U_puddles(L; U_disorder="gaussian", U=4.0, VI=4.0, λ=0.7, V0=0.0, impurity=0.075, rz=0.7, n=1000, seed=rand(Int64), signs=[1])
		c_U =make_U_params(U, L, U_disorder; U_puddle_params=(V0, VI, λ, rz), impurity=impurity, signs=signs, seed=seed)
        (U_mean, U_mean_std), (U_max, U_max_std), (U_min, U_min_std) = U_puddles_stat(L, n; U_disorder=U_disorder, U=U, VI=VI, λ=λ, V0=V0, impurity_ratio=impurity, rz=rz, stds=true, signs=signs)


		if U_disorder =="gaussian"
			tit = "L=$L, U=$(U), V=$impurity \n <U>=$(round(U_mean, sigdigits=2)) ± $(round(U_mean_std, sigdigits=1)), max=$(round(U_max, sigdigits=2)) ± $(round(U_max_std, sigdigits=1)), min=$(round(U_min, sigdigits=2)) ± $(round(U_min_std, sigdigits=1))"
		elseif U_disorder == "puddles"
			tit = "L=$L, VI=$VI, λ=$λ, rz=$rz, impurity=$impurity \n <U>=$(round(U_mean, sigdigits=2)) ± $(round(U_mean_std, sigdigits=1)), max=$(round(U_max, sigdigits=2)) ± $(round(U_max_std, sigdigits=1)), min=$(round(U_min, sigdigits=2)) ± $(round(U_min_std, sigdigits=1))"
		else
			tit = "L=$L"
		end

        fig = heatmap(reshape(c_U, L, L), title=tit, xlabel="x", ylabel="y", titlefontsize=10)
        return fig
    end

    """
        U_puddles_stat(L, n; U=4.0, VI=12.0, λ=0.7, V0=0.0, impurity_ratio=0.075, rz=0.7,
                       stds=false, U_disorder="gaussian", signs=[1])

    Mean over `n` random L×L maps (`make_U_params`) of the map's mean, max and min of U.
    With `stds=true` each comes with its standard error: ((mean, err), (max, err), (min, err)).
    """
    function U_puddles_stat(L, n; U=4.0, VI=12.0, λ=0.7, V0=0.0, impurity_ratio=0.075, rz=0.7, stds=false, U_disorder="gaussian", signs=[1])
        U_mean = zeros(Float64, n)
        U_max = zeros(Float64, n)
        U_min = zeros(Float64, n)

        for i in 1:n
            c_U = make_U_params(U, L, U_disorder; U_puddle_params=(V0, VI, λ, rz), impurity=impurity_ratio, signs=signs)
            U_mean[i] = mean(c_U)
            U_max[i] = maximum(c_U)
            U_min[i] = minimum(c_U)
        end
        U_mean_mean= mean(U_mean)
        U_mean_std = std(U_mean)/sqrt(n)
        U_max_mean = mean(U_max)
        U_max_std = std(U_max)/sqrt(n)
        U_min_mean = mean(U_min)
        U_min_std = std(U_min)/sqrt(n)

        if stds
            return (U_mean_mean, U_mean_std), (U_max_mean, U_max_std), (U_min_mean, U_min_std)
        else
            return (U_mean_mean, U_max_mean, U_min_mean)
        end    
    end
end

begin
	"""
	    extract_plot_digitizer(path) -> Vector{Tuple{Vector, Vector}}

	Read a WebPlotDigitizer JSON export (curves digitized from a paper's figure) and return
	each dataset as (x, y), sorted by x. Used to compare with published results.
	"""
	function extract_plot_digitizer(path::String)
		full_data = JSON.parsefile(path)
		datasetColl = full_data["datasetColl"]
		datas = []
		for i in eachindex(datasetColl)
			data = datasetColl[i]["data"]
			name = String(datasetColl[i]["name"])
			x = Float64[]
			y = Float64[]
			for j in eachindex(data)
				values = data[j]["value"]
				append!(x, values[1])
				append!(y, values[2])
			end
			sorted_indices = sortperm(x)
			x = x[sorted_indices]
    		y = y[sorted_indices]
			push!(datas, (x, y))
		end

		return datas
	end
end