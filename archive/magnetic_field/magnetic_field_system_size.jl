

function scalar_basis_size(L, Q)
    if L < Q
        new_L = rem(Q, L)
        if new_L == 0
            return 1
        end
        scalar_basis_size(new_L, Q)
    end

    return div(L, gcd(L, Q))
end

function basis_size(L, Q)

    return scalar_basis_size(L, Q) * scalar_basis_size(L+1, Q)
end

function print_basis_size(max_L; min_L=1)
    for L in min_L:max_L
        max_Q = div(L*(L+1) ,2)
        for Q in 1:max_Q
             size = basis_size(L, Q)
             println("L: $L , N: $(L * (L+1)), Q: $Q, Basis Size: $size")
        end
    end
end

function system_sizes(L; max_Q=0)
    if max_Q == 0
        max_Q = div(L*(L+1) ,2)
    end
    return  [basis_size(L, Q) for Q in 1:max_Q]
end


function L_Lps_even(max_L)
    # check up to max_L, which L values have even L(L+1)
    return [iseven(L*(L+1)) for L in 1:max_L]
end