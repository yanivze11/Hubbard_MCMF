include(joinpath(@__DIR__, "..", "..", "MC_Hubburd.jl"))
include(joinpath(@__DIR__, "..", "..", "sim_num_control.jl"))

function check_diag_symmetry(name::String, values, errors, axis_values;
							xx_idx::Int=1, yy_idx::Int=4, nsigma_tol::Float64=3.0)
	@assert length(values) == length(axis_values) "$(name): values length mismatch"
	@assert length(errors) == length(axis_values) "$(name): errors length mismatch"

	println("Checking xx≈yy symmetry for $(name) with tolerance $(nsigma_tol)σ")
	for i in eachindex(axis_values)
		v = values[i]
		e = errors[i]
		@assert v !== nothing "$(name): missing value at index $i"
		@assert e !== nothing "$(name): missing error at index $i"
		@assert length(v) >= yy_idx "$(name): tensor value must have at least $(yy_idx) components"
		@assert length(e) >= yy_idx "$(name): tensor error must have at least $(yy_idx) components"

		rho_xx = Float64(v[xx_idx])
		rho_yy = Float64(v[yy_idx])
		err_xx = Float64(e[xx_idx])
		err_yy = Float64(e[yy_idx])

		@assert isfinite(rho_xx) && isfinite(rho_yy) "$(name): non-finite diagonal value at index $i"

		# If errors are unavailable/invalid, skip n-sigma check but report the mismatch.
		if !(isfinite(err_xx) && isfinite(err_yy))
			println("  i=$(i), axis=$(axis_values[i]): xx=$(rho_xx), yy=$(rho_yy), err invalid -> skipped nσ check")
			continue
		end

		sigma = sqrt(err_xx^2 + err_yy^2)
		delta = abs(rho_xx - rho_yy)
		nsigma = sigma > 0 ? delta / sigma : (delta == 0 ? 0.0 : Inf)
		println("  i=$(i), axis=$(axis_values[i]): xx=$(rho_xx), yy=$(rho_yy), Δ=$(delta), nσ=$(nsigma)")
		@assert nsigma <= nsigma_tol "$(name): xx vs yy mismatch above tolerance at index $i (nσ=$(nsigma), tol=$(nsigma_tol))"
	end
end

function quick_test_dc_resistivity_all_dw()
	dw_values = [0.01, 0.02, 0.05, 0.1]
	delta_values_fixed = [0.1, 0.2, 0.3]

	data = running_simulation_once(
		4, 4,                          # Lx, Ly
		1,                             # L_SC
		2.0, 1.0, 0.0,                 # U, t, V
		12345,                         # seed
		0.0, 2.0, 0.0,                 # mu, beta, B
		1000, 200, 1,                       # num_measure, num_thermal, measuring_interval
		[-2.0, -1.0, 0.0, 1.0, 2.0],   # w_list
		"uniform", "uniform", "K0", "s-wave", # initial_guess, U_disorder, diag_method, pairing_symmetry
		[0.1, 0.2, 0.3],               # w_conductivity_range (avoid w=0 for derived regular conductivity)
		1.0;                           # U_impurity
		savedata=false,
		saveplots=false,
		date_folder=false,
		numbered=false,
		dw_values=dw_values,
		delta_values_fixed=delta_values_fixed
	)

	means = get_means(data)
	@assert haskey(means, "resistivity_dc_tensor") "resistivity_dc_tensor was not computed"

	dc_res = means["resistivity_dc_tensor"]
	vals = dc_res["value"]
	errs = dc_res["error"]

	@assert length(vals) == length(dw_values) "Expected $(length(dw_values)) dw entries, got $(length(vals))"
	@assert length(errs) == length(dw_values) "Error vector length mismatch for resistivity_dc_tensor"

	println("DW values: ", dw_values)
	println("resistivity_dc_tensor entries: ", length(vals))

	for i in eachindex(dw_values)
		@assert vals[i] !== nothing "Missing resistivity tensor at dw index $i (dw=$(dw_values[i]))"
		@assert errs[i] !== nothing "Missing resistivity tensor error at dw index $i (dw=$(dw_values[i]))"
		@assert length(vals[i]) == 4 "resistivity tensor must have 4 components at dw index $i"
		@assert length(errs[i]) == 4 "resistivity tensor error must have 4 components at dw index $i"
		println("dw=$(dw_values[i]) -> rho=$(vals[i]) ; err=$(errs[i])")
	end
	check_diag_symmetry("resistivity_dc_tensor", vals, errs, dw_values)

	@assert haskey(means, "resistivity_tensor_fixed") "resistivity_tensor_fixed was not computed"
	fixed_res = means["resistivity_tensor_fixed"]
	fixed_vals = fixed_res["value"]
	fixed_errs = fixed_res["error"]
	@assert length(fixed_vals) == length(delta_values_fixed) "Expected $(length(delta_values_fixed)) delta entries, got $(length(fixed_vals))"
	@assert length(fixed_errs) == length(delta_values_fixed) "Error vector length mismatch for resistivity_tensor_fixed"
	println("delta_values_fixed: ", delta_values_fixed)
	println("resistivity_tensor_fixed entries: ", length(fixed_vals))
	for i in eachindex(delta_values_fixed)
		@assert fixed_vals[i] !== nothing "Missing fixed resistivity tensor at delta index $i (delta=$(delta_values_fixed[i]))"
		@assert fixed_errs[i] !== nothing "Missing fixed resistivity tensor error at delta index $i (delta=$(delta_values_fixed[i]))"
		@assert length(fixed_vals[i]) == 4 "fixed resistivity tensor must have 4 components at delta index $i"
		@assert length(fixed_errs[i]) == 4 "fixed resistivity tensor error must have 4 components at delta index $i"
		println("delta=$(delta_values_fixed[i]) -> rho=$(fixed_vals[i]) ; err=$(fixed_errs[i])")
	end
	check_diag_symmetry("resistivity_tensor_fixed", fixed_vals, fixed_errs, delta_values_fixed)

	println("PASS: resistivity tensors were computed and xx≈yy checks passed for dw and fixed-delta.")
	return means
end

if abspath(PROGRAM_FILE) == @__FILE__
	quick_test_dc_resistivity_all_dw()
end

