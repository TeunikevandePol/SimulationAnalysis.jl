"""
    find_intermediate_scattering_function(s::Simulation; kmin=7.0, kmax=7.4, kfactor=1)

Calculates the coherent intermediate scattering function `F(k, t)` for a simulation, averaged over a shell in k-space.

This is a convenience function that first constructs the k-space and density modes, and then computes `F(k, t)`.
The function is defined as: `F(k, t) = (1/N) * < Σ_{i,j} exp(i * k ⋅ (r_i(t) - r_j(0))) >`.
The average is taken over time origins and over k-vectors with magnitudes `k` such that `kmin < k < kmax`.

# Arguments
- `s::Simulation`: The simulation data.
- `kmin::Float64=7.0`: The minimum magnitude of the k-vectors to be included in the average.
- `kmax::Float64=7.4`: The maximum magnitude of the k-vectors to be included in the average.
- `kfactor::Int=1`: The resolution factor for the k-space grid.

# Returns
- `Fk`: The intermediate scattering function. For a `SingleComponentSimulation`, this is a `Vector{Float64}`. For a `MultiComponentSimulation`, this is a `Matrix{Vector{Float64}}`.
"""
function find_intermediate_scattering_function(s::Simulation; kmin=7.0, kmax=7.4, kfactor=1)
    kspace = construct_k_space(s, (kmin, kmax); kfactor=kfactor, negative=true, rectangular=false)
    ρkt = find_density_modes(s, kspace; verbose=false)
    F = find_intermediate_scattering_function(s, kspace, ρkt; kmin=kmin, kmax=kmax)
    return F
end

"""
    find_intermediate_scattering_function(s::Simulation, kspace::KSpace, ρkt, k_sample_array::AbstractVector; k_binwidth=0.1)

Calculates the coherent intermediate scattering function `F(k, t)` for a list of specified `k` values.

For each `k` in `k_sample_array`, this function computes `F(k, t)` by averaging over a k-shell of width `k_binwidth` centered at `k`.

# Arguments
- `s::Simulation`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `ρkt`: The pre-computed density modes (`SingleComponentDensityModes` or `MultiComponentDensityModes`).
- `k_sample_array::AbstractVector`: A vector of k-magnitudes for which to compute `F(k, t)`.
- `k_binwidth::Float64=0.1`: The width of the k-shell to average over for each value in `k_sample_array`.

# Returns
- `F_array::Vector`: A vector where `F_array[i]` is the `F(k,t)` corresponding to `k_sample_array[i]`.
"""
function find_intermediate_scattering_function(s::Simulation, kspace::KSpace, ρkt, k_sample_array::AbstractVector; k_binwidth=0.1)
    F_array = []
    for (ik, k) in enumerate(k_sample_array)
        kmin = k - k_binwidth/2
        kmax = k + k_binwidth/2
        push!(F_array, find_intermediate_scattering_function(s, kspace, ρkt; kmin=kmin, kmax=kmax))
    end
    return F_array
end

"""
    find_intermediate_scattering_function(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace, ρkt::SingleComponentDensityModes; kmin=0.0, kmax=10.0^10.0)

Calculates the coherent intermediate scattering function `F(k, t)` for a single-component simulation.

This is the main implementation that computes `F(k, t) = (1/N) * <ρ(k, t) ρ*(-k, 0)>` by correlating the pre-computed density modes `ρkt`.
The average is performed over time origins and k-vectors within the magnitude range `[kmin, kmax]`.

# Arguments
- `s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `ρkt::SingleComponentDensityModes`: The pre-computed density modes.
- `kmin::Float64=0.0`: The minimum magnitude of k-vectors to include in the average.
- `kmax::Float64=10.0^10.0`: The maximum magnitude of k-vectors to include in the average.

# Returns
- `Fk::Vector{Float64}`: A vector containing `F(k, t)` for each time delay `Δt` in `s.dt_array`.
"""
function find_intermediate_scattering_function(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace, ρkt::SingleComponentDensityModes; kmin=0.0, kmax=10.0^10.0)
    Ndt = length(s.dt_array)
    Fk = zeros(Ndt)
    real_correlation_function!(Fk, ρkt.Re, ρkt.Im, ρkt.Re, ρkt.Im, kspace, s.dt_array, s.t1_t2_pair_array, kmin, kmax)
    return Fk ./ s.N
end

"""
    find_intermediate_scattering_function(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace, ρkt::MultiComponentDensityModes; kmin=0.0, kmax=10.0^10.0)

Calculates the partial intermediate scattering functions `F_αβ(k, t)` for a multi-component simulation.

This function computes `F_αβ(k, t) = (1/N) * <ρ_α(k, t) ρ_β*(-k, 0)>` by correlating the pre-computed density modes `ρkt` for species `α` and `β`.
The average is performed over time origins and k-vectors within the magnitude range `[kmin, kmax]`.

# Arguments
- `s::Union{MultiComponentSimulation,MCSPVSimulation}`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `ρkt::MultiComponentDensityModes`: The pre-computed density modes for all species.
- `kmin::Float64=0.0`: The minimum magnitude of k-vectors to include in the average.
- `kmax::Float64=10.0^10.0`: The maximum magnitude of k-vectors to include in the average.

# Returns
- `Fk::Matrix{Vector{Float64}}`: A matrix of vectors, where `Fk[α, β]` is the partial intermediate scattering function `F_αβ(k, t)`.
"""
function find_intermediate_scattering_function(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace, ρkt::MultiComponentDensityModes; kmin=0.0, kmax=10.0^10.0)
    N_species = s.N_species
    Ndt = length(s.dt_array)
    Fk = [zeros(Ndt) for α=1:N_species, β=1:N_species]
    for α=1:N_species
        for β = 1:N_species
            real_correlation_function!(Fk[α,β], ρkt.Re[α], ρkt.Im[α], ρkt.Re[β], ρkt.Im[β], kspace, s.dt_array, s.t1_t2_pair_array, kmin, kmax)
            # Fk[β, α] .= Fk[α,β]
        end
    end
    return Fk ./ s.N
end

"""
    find_self_intermediate_scattering_function(s::Simulation; kmin=7.0, kmax=7.4, kfactor=1)

Calculates the self-intermediate scattering function `Fs(k, t)`, averaged over a shell in k-space.

This is a convenience function that first constructs the k-space and then computes `Fs(k, t)`.
The function is defined as: `Fs(k, t) = (1/N) * <Σ_j exp(i * k ⋅ (r_j(t) - r_j(0)))>`.
The average is taken over time origins and over k-vectors with magnitudes `k` such that `kmin < k < kmax`.
This function is only implemented for `SingleComponentSimulation`.

# Arguments
- `s::Simulation`: The simulation data.
- `kmin::Float64=7.0`: The minimum magnitude of the k-vectors to be included in the average.
- `kmax::Float64=7.4`: The maximum magnitude of the k-vectors to be included in the average.
- `kfactor::Int=1`: The resolution factor for the k-space grid.

# Returns
- `Fk::Vector{Float64}`: A vector containing `Fs(k, t)` for each time delay `Δt` in `s.dt_array`.
- `Fks_per_particle::Matrix{Float64}`: A `(Ndt, N)` matrix containing the self-intermediate scattering function for each particle.
"""
function find_self_intermediate_scattering_function(s::Simulation; kmin=7.0, kmax=7.4, kfactor=1)
    kspace = construct_k_space(s, (kmin, kmax); kfactor=kfactor, negative=true, rectangular=false)
    F = find_self_intermediate_scattering_function(s, kspace; kmin=kmin, kmax=kmax)
    return F
end

"""
    find_self_intermediate_scattering_function(s::Simulation, kspace::KSpace, k_sample_array::AbstractVector; k_binwidth=0.1)

Calculates the self-intermediate scattering function `Fs(k, t)` for a list of specified `k` values.

For each `k` in `k_sample_array`, this function computes `Fs(k, t)` by averaging over a k-shell of width `k_binwidth` centered at `k`.
This function is only implemented for `SingleComponentSimulation`.

# Arguments
- `s::Simulation`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `k_sample_array::AbstractVector`: A vector of k-magnitudes for which to compute `Fs(k, t)`.
- `k_binwidth::Float64=0.1`: The width of the k-shell to average over for each value in `k_sample_array`.

# Returns
- `F_array::Vector`: A vector of tuples `(Fk, Fks_per_particle)` for each `k` in `k_sample_array`.
"""
function find_self_intermediate_scattering_function(s::Simulation, kspace::KSpace, k_sample_array::AbstractVector; k_binwidth=0.1)
    F_array = []
    for (ik, k) in enumerate(k_sample_array)
        # if verbose
        #     println("Computing Fs for k = $k")
        # end
        kmin = k - k_binwidth/2
        kmax = k + k_binwidth/2
        push!(F_array, find_self_intermediate_scattering_function(s, kspace; kmin=kmin, kmax=kmax))
    end
    return F_array
end

"""
    find_self_intermediate_scattering_function(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace; kmin=0.0, kmax=10.0^10.0)

Calculates the self-intermediate scattering function `Fs(k, t)` for a single-component simulation.

This is the main implementation that computes `Fs(k, t) = (1/N) * <Σ_j exp(i * k ⋅ (r_j(t) - r_j(0)))>`.
The calculation is done without pre-calculating density modes to avoid large memory allocation.
The average is performed over time origins and k-vectors within the magnitude range `[kmin, kmax]`.

# Arguments
- `s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}`: The simulation data.
- `kspace::KSpace`: The pre-computed k-space.
- `kmin::Float64=0.0`: The minimum magnitude of k-vectors to include in the average.
- `kmax::Float64=10.0^10.0`: The maximum magnitude of k-vectors to include in the average.

# Returns
- `Fk::Vector{Float64}`: A vector containing `Fs(k, t)` for each time delay `Δt` in `s.dt_array`.
- `Fks_per_particle::Matrix{Float64}`: A `(Ndt, N)` matrix containing the self-intermediate scattering function for each particle.
"""
function find_self_intermediate_scattering_function(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace; kmin=0.0, kmax=10.0^10.0)
    kmask = (kmin .< kspace.k_lengths .< kmax)
    k_lengths = kspace.k_lengths[kmask]
    k_array = kspace.k_array[:, kmask]
    r_array = s.r_array
    N = size(r_array, 2)
    Nk = length(k_lengths)
    Ndt = length(s.dt_array)
    Nt = length(s.t_array)
    coskr = zeros(Nt, Nk)
    sinkr = zeros(Nt, Nk)
    Fks_per_particle = zeros(Ndt, N)
    for particle = 1:N
        if s.Ndims == 3
            @turbo for ik = 1:Nk
                kx = k_array[1, ik]
                ky = k_array[2, ik]
                kz = k_array[3, ik]
                for it = 1:Nt
                    r1x = r_array[1, particle, it]
                    r1y = r_array[2, particle, it]
                    r1z = r_array[3, particle, it]
                    rk1 = kx*r1x + ky*r1y + kz*r1z
                    sinrk1, cosrk1 = sincos(rk1)
                    coskr[it, ik] = cosrk1
                    sinkr[it, ik] = sinrk1
                end
            end
        elseif s.Ndims == 2
            @turbo for ik = 1:Nk
                kx = k_array[1, ik]
                ky = k_array[2, ik]
                for it = 1:Nt
                    r1x = r_array[1, particle, it]
                    r1y = r_array[2, particle, it]
                    rk1 = kx*r1x + ky*r1y
                    sinrk1, cosrk1 = sincos(rk1)
                    coskr[it, ik] = cosrk1
                    sinkr[it, ik] = sinrk1
                end
            end
        else
            error("Only 2D and 3D simulations are supported")
        end
        for iδt in eachindex(s.dt_array)
            fkspartialidt = 0.0
            pairs_idt = s.t1_t2_pair_array[iδt]
            Npairs = size(pairs_idt, 1)
            @turbo for ipair = 1:Npairs
                t1 = pairs_idt[ipair, 1]
                t2 = pairs_idt[ipair, 2]
                for ik = 1:Nk
                    fkspartialidt += coskr[t2, ik]*coskr[t1, ik] + sinkr[t2, ik]*sinkr[t1, ik]
                end
            end
            Fks_per_particle[iδt, particle] += fkspartialidt
        end
    end

    for iδt in eachindex(s.dt_array)
        pairs_idt = s.t1_t2_pair_array[iδt]
        Npairs = size(pairs_idt, 1)
        Fks_per_particle[iδt, :] ./= Nk*Npairs
    end
    Fk = sum(Fks_per_particle; dims=2)[:] / N
    return Fk, Fks_per_particle
end
