using JLD2, Dates
f = "gaussian_model_results/sim_info_gaussian_model.jld2"
si = JLD2.load(f)
note = "DOS-focused run: DOS,n,Ps,chi_spin,chi_charge,Ps_l,Phase_Corr,Delta_* are measured; " *
       "transport/spectral observables (Kx,Ky,Lxx_*/Lyy_*,conductivity_*,stiffness*,spectral A/BZ) " *
       "were SKIPPED and are stored as zeros — do NOT use them as real data."
for (k,e) in si
    k=="updated_at" && continue
    (e isa AbstractDict) || continue
    e["gaussian_fast_dos_mode"] = true
    e["gaussian_measurement_note"] = note
    si[k] = e
end
si["updated_at"] = Dates.now()
JLD2.save(f, si)
println("stamped provenance on $(count(k->k!="updated_at", collect(keys(si)))) entries")
