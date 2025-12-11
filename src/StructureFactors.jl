# function _find_S2(kspace::KSpace)
#     println("Calculating Structure factor")
#     Reρkt = kspace.Reρkt
#     Imρkt = kspace.Reρkt
#     N_timesteps = size(Reρkt)[1]
#     @time @tullio S2[iq] := Reρkt[t, iq]^2 + Imρkt[t, iq]^2
#     S2 ./= N_timesteps*kspace.N
#     return kspace.k_lengths, S2
# end


# function find_S2(kspace, k_sample_array; dq=0.1)  
#     k_lengths, S2 = _find_S2(kspace)
#     println("Binning the structure factor")
#     Nk_sample = length(k_sample_array)
#     S2_binned = zeros(Nk_sample)
#     counts = zeros(Nk_sample)
#     @time for i ∈ 1:length(S2)
#         k_i = k_lengths[i]
#         for i_sample ∈ 1:Nk_sample
#             k_sample = k_sample_array[i_sample]
#             if abs(k_i - k_sample) < dq
#                 S2_binned[i_sample] += S2[i]
#                 counts[i_sample] += 1
#             end
#         end
#     end
#     S2_binned[counts .> 0] ./= counts[counts .> 0]
#     return S2_binned
# end

"""
    find_structure_factor(s::Simulation; kmin=7.0, kmax=7.4, kfactor=1)

Calculates the static structure factor `S(k)` for a simulation, averaged over a shell in k-space.

This is a convenience function that first constructs the k-space and density modes, and then computes `S(k)`.
The function is defined as: `S(k) = (1/N) * <|Σ_j exp(i * k ⋅ r_j)|^2>`.
The average is taken over time and over k-vectors with magnitudes `k` such that `kmin < k < kmax`.

# Arguments
- `s::Simulation`: The simulation data.
- `kmin::Float64=7.0`: The minimum magnitude of the k-vectors to be included in the average.
- `kmax::Float64=7.4`: The maximum magnitude of the k-vectors to be included in the average.
- `kfactor::Int=1`: The resolution factor for the k-space grid.

# Returns
- `Sk`: The structure factor. For a `SingleComponentSimulation`, this is a `Float64`. For a `MultiComponentSimulation`, this is a `Matrix{Float64}`.
"""
function find_structure_factor(s::Simulation; kmin=7.0, kmax=7.4, kfactor=1)
    kspace = construct_k_space(s, (kmin, kmax); kfactor=kfactor, negative=true, rectangular=false)
    ρkt = find_density_modes(s, kspace; verbose=false)
    F = find_structure_factor(s, kspace, ρkt; kmin=kmin, kmax=kmax)
    return F
end

"""
    find_structure_factor(s::Simulation, kspace::KSpace, ρkt::AbstractDensityModes, k_sample_array::AbstractVector; k_binwidth=0.1)

Calculates the static structure factor `S(k)` for a list of specified `k` values.

For each `k` in `k_sample_array`, this function computes `S(k)` by averaging over a k-shell of width `k_binwidth` centered at `k`.

# Arguments
- `s::Simulation`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `ρkt::AbstractDensityModes`: The pre-computed density modes.
- `k_sample_array::AbstractVector`: A vector of k-magnitudes for which to compute `S(k)`.
- `k_binwidth::Float64=0.1`: The width of the k-shell to average over for each value in `k_sample_array`.

# Returns
- `S_array::Vector`: A vector where `S_array[i]` is the `S(k)` corresponding to `k_sample_array[i]`.
"""
function find_structure_factor(s::Simulation, kspace::KSpace, ρkt::AbstractDensityModes, k_sample_array::AbstractVector; k_binwidth=0.1)
    S_array = []
    for (ik, k) in enumerate(k_sample_array)
        kmin = k - k_binwidth/2
        kmax = k + k_binwidth/2
        push!(S_array, find_structure_factor(s, kspace, ρkt; kmin=kmin, kmax=kmax))
    end
    return S_array
end

"""
    find_structure_factor(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace, ρkt::SingleComponentDensityModes; kmin=0.0, kmax=10.0^10.0)

Calculates the static structure factor `S(k)` for a single-component simulation.

This is the main implementation that computes `S(k) = (1/N) * <|ρ(k)|^2>` from the pre-computed density modes `ρkt`.
The average is performed over time and k-vectors within the magnitude range `[kmin, kmax]`.

# Arguments
- `s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `ρkt::SingleComponentDensityModes`: The pre-computed density modes.
- `kmin::Float64=0.0`: The minimum magnitude of k-vectors to include in the average.
- `kmax::Float64=10.0^10.0`: The maximum magnitude of k-vectors to include in the average.

# Returns
- `Sk::Float64`: The value of the structure factor `S(k)`.
"""
function find_structure_factor(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace, ρkt::SingleComponentDensityModes; kmin=0.0, kmax=10.0^10.0)
    Sk = real_static_correlation_function(ρkt.Re, ρkt.Im, ρkt.Re, ρkt.Im, kspace, kmin, kmax)
    return Sk / s.N
end


"""
    find_structure_factor(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace, ρkt::MultiComponentDensityModes; kmin=0.0, kmax=10.0^10.0)

Calculates the partial static structure factors `S_αβ(k)` for a multi-component simulation.

This function computes `S_αβ(k) = (1/N) * <ρ_α(k) ρ_β*(-k)>` from the pre-computed density modes `ρkt` for species `α` and `β`.
The average is performed over time and k-vectors within the magnitude range `[kmin, kmax]`.

# Arguments
- `s::Union{MultiComponentSimulation,MCSPVSimulation}`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `ρkt::MultiComponentDensityModes`: The pre-computed density modes for all species.
- `kmin::Float64=0.0`: The minimum magnitude of k-vectors to include in the average.
- `kmax::Float64=10.0^10.0`: The maximum magnitude of k-vectors to include in the average.

# Returns
- `Sk::Matrix{Float64}`: A matrix where `Sk[α, β]` is the partial structure factor `S_αβ(k)`.
"""
function find_structure_factor(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace, ρkt::MultiComponentDensityModes; kmin=0.0, kmax=10.0^10.0)
    N_species = s.N_species
    Sk = zeros(N_species, N_species)
    # for α=1:N_species
    #     for β = α:N_species
    #         Sk[β, α] = real_static_correlation_function(ρkt.Re[α], ρkt.Im[α], ρkt.Re[β], ρkt.Im[β], kspace, kmin, kmax)
    #         if α != β
    #             Sk[α, β] = Sk[β, α]
    #         end
    #     end
    # end
    for α=1:N_species, β=1:N_species
        Sk[α,β] = real_static_correlation_function(ρkt.Re[α], ρkt.Im[α], ρkt.Re[β], ρkt.Im[β], kspace, kmin, kmax)
    end
    return Sk / s.N
end



"""
    _find_S4_offdiagonal(s, kspace, ρkt, costheta12_bounds, costheta13_bounds, phi23_bounds, q_arr, Sq_binned, maxsamples, maxq, iq1_set, iq2_set, iq3_set)

Calculates the off-diagonal part of the four-point structure factor.
"""
function _find_S4_offdiagonal(s::SingleComponentSimulation, kspace::KSpace, ρkt::AbstractDensityModes, costheta12_bounds, costheta13_bounds, phi23_bounds, q_arr, Sq_binned, maxsamples, maxq, iq1_set, iq2_set, iq3_set)
    N_timesteps = size(ρkt.Re, 1)
    q_lengths = kspace.k_lengths
    q_array = kspace.k_array
    Nq = length(q_lengths)
    S_interp = Spline1D(q_arr, Sq_binned)
    dqx = 2π/s.box_sizes[1]*kspace.kfactor
    dqy = 2π/s.box_sizes[2]*kspace.kfactor
    dqz = 2π/s.box_sizes[3]*kspace.kfactor
    @assert s.Ndims == 3
    S₄ = 0.0
    S₄_conv = 0.0
    total_q_samples = 0

    Reρkt = ρkt.Re
    Imρkt = ρkt.Im

    for iq1 in iq1_set
        q1x = q_array[1, iq1]
        q1y = q_array[2, iq1]
        q1z = q_array[3, iq1]
        q1 = q_lengths[iq1]
        q1_indices = (round(Int, q1x/dqx), round(Int, q1y/dqy), round(Int, q1z/dqz))
        for iq2 in iq2_set
            q2x = q_array[1, iq2]
            q2y = q_array[2, iq2]
            q2z = q_array[3, iq2]
            q2 = q_lengths[iq2]
            if q1x == -q2x && q1y == -q2y && q1z == -q2z # no diagonal terms
                continue
            end
            q2_indices = (round(Int, q2x/dqx), round(Int, q2y/dqy), round(Int, q2z/dqz))
            q1_in_q2 = q1x*q2x + q1y*q2y + q1z*q2z
            costheta12 = q1_in_q2 / (q1*q2)
            if costheta12 < costheta12_bounds[1] || costheta12 > costheta12_bounds[2] 
                continue
            end
            if total_q_samples >= maxsamples
                break
            end
            for iq3 in iq3_set
                if total_q_samples >= maxsamples
                    break
                end
                q3x = q_array[1, iq3]
                q3y = q_array[2, iq3]
                q3z = q_array[3, iq3]
                q3 = q_lengths[iq3]
                if abs(q1x+q2x+q3x) > maxq || abs(q1y+q2y+q3y)>maxq || abs(q1z+q2z+q3z)>maxq # q4 needs to be part of the set of vectors for which we calculated the density modes
                    error("excluding vectors that are too large")  
                    continue
                end
                q3_indices = (round(Int, q3x/dqx), round(Int, q3y/dqy), round(Int, q3z/dqz))
                q1_in_q3 = q1x*q3x + q1y*q3y + q1z*q3z
                costheta13 = q1_in_q3 / (q1*q3)
                if costheta13 < costheta13_bounds[1] || costheta13 > costheta13_bounds[2]
                    continue
                end
                if iq1 == iq3 || iq2 == iq3
                    continue
                end
                q2_in_q3 = q2x*q3x + q2y*q3y + q2z*q3z
                q1² = q1*q1
                q2² = q2*q2
                q3² = q3*q3
                cosphi23num = q1²*q2_in_q3 - q1_in_q2*q1_in_q3
                cosphi23 = 0.0
                if abs(cosphi23num) < 10^-14 # if the numerator is zero
                    cosphi23 = 0.0
                elseif iq2 == iq3 || abs(q1*q2 - abs(q1_in_q2)) < 10^-10  || abs(q1*q3 - abs(q1_in_q3)) < 10^-10 #if q2==q3 then we cannot calculate phi23 and therefore set it to 0.0
                    cosphi23 = 1.0 
                else
                    cosphi23 = cosphi23num/sqrt((q1²*q2² - q1_in_q2^2)*(q1²*q3² - q1_in_q3^2)) # this is the angle between the projections of q2 and q3 on the plane perpendicular to q1
                end
                if 1.0 < cosphi23 < 1.0 + 1e-5 # if a numerical error occurs to make the cos(phi)>1.0, set it to phi = 0.0
                    cosphi23 = 1.0
                elseif -1.0 - 1e-5 < cosphi23 < -1.0
                    cosphi23 = -1.0
                end
                phi23 = acos(cosphi23)
                if phi23 < phi23_bounds[1] || phi23 > phi23_bounds[2]
                    continue
                end
                q4_indices1 = q1_indices[1] + q2_indices[1] + q3_indices[1]
                q4_indices2 = q1_indices[2] + q2_indices[2] + q3_indices[2]
                q4_indices3 = q1_indices[3] + q2_indices[3] + q3_indices[3]

                iq4 = kspace.cartesian_to_linear[q4_indices1, q4_indices2, q4_indices3]
                q4x = q_array[1, iq4]
                q4y = q_array[2, iq4]
                q4z = q_array[3, iq4]
                q4 = sqrt(q4x^2+q4y^2+q4z^2)
                total_q_samples += 1
                @inbounds for t = 1:N_timesteps
                    Reρ1 = Reρkt[t, iq1]
                    Imρ1 = Imρkt[t, iq1]
                    Reρ2 = Reρkt[t, iq2]
                    Imρ2 = Imρkt[t, iq2]
                    Reρ3 = Reρkt[t, iq3]
                    Imρ3 = Imρkt[t, iq3]
                    Reρ4 = Reρkt[t, iq4]
                    Imρ4 = -Imρkt[t, iq4]
                    S₄ += Reρ1*Reρ2*Reρ3*Reρ4 - Reρ1*Reρ2*Imρ3*Imρ4 - Reρ1*Imρ2*Reρ3*Imρ4 - Imρ1*Reρ2*Reρ3*Imρ4 - Reρ1*Imρ2*Imρ3*Reρ4 - Imρ1*Reρ2*Imρ3*Reρ4 - Imρ1*Imρ2*Reρ3*Reρ4 + Imρ1*Imρ2*Imρ3*Imρ4 
                    #    Re((Reρ1 + im*Imρ1)*(Reρ2 + im*Imρ2)*(Reρ3 + im*Imρ3)*(Reρ4 + im*Imρ4))
                end
                # calculate conv. approx.:
                abs_q1_plus_q2 = sqrt((q1x+q2x)^2 + (q1y+q2y)^2 + (q1z+q2z)^2)
                abs_q1_plus_q3 = sqrt((q1x+q3x)^2 + (q1y+q3y)^2 + (q1z+q3z)^2)
                abs_q2_plus_q3 = sqrt((q2x+q3x)^2 + (q2y+q3y)^2 + (q2z+q3z)^2)
                S_k1 = S_interp(q1)
                S_k2 = S_interp(q2)
                S_k3 = S_interp(q3)
                S_k4 = S_interp(q4)
                S_k12 = S_interp(abs_q1_plus_q2)
                S_k13 = S_interp(abs_q1_plus_q3)
                S_k23 = S_interp(abs_q2_plus_q3)
                S₄_conv += S_k1*S_k2*S_k3*S_k4*(S_k12 + S_k13 + S_k23 - 2.0)

            end
        end
    end
    if total_q_samples == 0
        # println("Warning: no suitable set of q-vectors was found, returning S4=0")
        return 0.0, 0.0
    end
    # println("Calculated S4 for $total_q_samples different sets of k-vectors")
    return S₄/N_timesteps/total_q_samples, S₄_conv/total_q_samples
end


"""
    _dispatch_S4(s, kspace, ρkt, Ntheta13, Nphi23, q_arr, Sq_binned; q1=7.2, dq1=0.1, q2=7.2 , dq2=0.1, costheta12=cos(0π/4), dcostheta12=0.05, q3=7.1, dq3=0.1, dcostheta13=0.05, dphi23=0.05π, maxsamples=10^10)

Dispatches the calculation of the four-point structure factor.
"""
function _dispatch_S4(s::Simulation, kspace::KSpace, ρkt::AbstractDensityModes, Ntheta13, Nphi23, q_arr, Sq_binned; q1=7.2, dq1=0.1, q2=7.2 , dq2=0.1, costheta12=cos(0π/4), dcostheta12=0.05, q3=7.1, dq3=0.1, dcostheta13=0.05, dphi23=0.05π, maxsamples=10^10)
    eps = 1e-5
    costheta13_array = collect(LinRange(-1+dcostheta13+eps , 1-dcostheta13-eps, Ntheta13))
    phi23_array = collect(LinRange(dphi23+eps, pi-dphi23-eps, Nphi23))
    S4 = zeros(Ntheta13, Nphi23)
    S4conv = zeros(Ntheta13, Nphi23)
    i_done = Atomic{Int}(0)
    println("calculating S4")
    k_lengths = kspace.k_lengths
    q1_bounds = (q1 - dq1, q1 + dq1)
    q2_bounds = (q2 - dq2, q2 + dq2)
    q3_bounds = (q3 - dq3, q3 + dq3)
    maxq = maximum(kspace.k_array)
    Nq = length(k_lengths)
    iq1_set = (1:Nq)[(q1_bounds[1] .< k_lengths .< q1_bounds[2])]
    iq2_set = (1:Nq)[(q2_bounds[1] .< k_lengths .< q2_bounds[2])]
    iq3_set = (1:Nq)[(q3_bounds[1] .< k_lengths .< q3_bounds[2])]

    @threads for i=1:length(costheta13_array)
        costheta13 = costheta13_array[i]
        @threads for j = 1:length(phi23_array)
            phi23 = phi23_array[j]

            costheta12_bounds = (costheta12-dcostheta12, costheta12+dcostheta12)
            costheta13_bounds = (costheta13-dcostheta13, costheta13+dcostheta13)
            phi23_bounds = (phi23-dphi23, phi23+dphi23)


            S4new, S4convnew = _find_S4_offdiagonal(s, kspace,ρkt, costheta12_bounds, costheta13_bounds, phi23_bounds, q_arr, Sq_binned, maxsamples, maxq, iq1_set, iq2_set, iq3_set) # calculates the offdiagonal S4 and conv approx for this particular choice of vectors
            S4[i,j] = S4new/s.N 
            S4conv[i,j] = S4convnew
            atomic_add!(i_done, 1)
            # println("done = $(i_done[])/$(length(costheta13_array)*length(phi23_array))")
        end
    end
    return costheta13_array, phi23_array, S4, S4conv
end

"""
    find_S4_offiagonal(s, kspace, ρkt, Ntheta13, Nphi23; q1=7.2, dq1=0.1, q2=7.2 , dq2=0.1, costheta12=cos(0π/4), dcostheta12=0.05, q3=7.1, dq3=0.1, dcostheta13=0.05, dphi23=0.05π, maxsamples=10^10)

Calculates the off-diagonal part of the four-point structure factor.
"""
function find_S4_offiagonal(s::SingleComponentSimulation, kspace::KSpace, ρkt::AbstractDensityModes, Ntheta13, Nphi23; q1=7.2, dq1=0.1, q2=7.2 , dq2=0.1, costheta12=cos(0π/4), dcostheta12=0.05, q3=7.1, dq3=0.1, dcostheta13=0.05, dphi23=0.05π, maxsamples=10^10)
    if maximum(kspace.k_array) < q1+q2+q3+dq1+dq2+dq3
        error("The wavevectors considered are too small to resolve the requested wave vector set. Maximal wavelength needs to be at least k=$(q1+q2+q3+dq1+dq2+dq3)")
    end 
    k_sample_array = collect(LinRange(minimum(kspace.k_lengths), maximum(kspace.k_lengths)/1.2, 100))[2:end]
    Sq_binned = find_structure_factor(s, kspace, ρkt, k_sample_array; k_binwidth=0.1)
    k_sample_array = k_sample_array[isfinite.(Sq_binned)]
    Sq_binned = Sq_binned[isfinite.(Sq_binned)] 
    _dispatch_S4(s, kspace, ρkt, Ntheta13, Nphi23, k_sample_array, Sq_binned; q1=q1, dq1=dq1, q2=q2, dq2=dq2, costheta12=costheta12, dcostheta12=dcostheta12, q3=q3, dq3=dq3, dcostheta13=dcostheta13, dphi23=dphi23, maxsamples=maxsamples)
end

