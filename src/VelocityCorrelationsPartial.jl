#########################################################################
## PARTIAL CURRENT MODES 

struct SingleComponentCurrentModesInt <: AbstractDensityModes
    Re::Array{Float64, 2}
    Im::Array{Float64, 2}
end

struct SingleComponentCurrentModesAct <: AbstractDensityModes
    Re::Array{Float64, 2}
    Im::Array{Float64, 2}
end

struct MultiComponentCurrentModesInt <: AbstractDensityModes
    Re::Vector{Array{Float64, 2}}
    Im::Vector{Array{Float64, 2}}
end

struct MultiComponentCurrentModesAct <: AbstractDensityModes
    Re::Vector{Array{Float64, 2}}
    Im::Vector{Array{Float64, 2}}
end

function _find_current_modes_partial!(Rej, Imj, r, Ftot, f, kspace; force_keyword="t")
    Ndim, N, N_timesteps = size(r)
    Nk = kspace.Nk
    k_array = kspace.k_array
    klengths = kspace.k_lengths

    if force_keyword == "t" # use total force
        F = Ftot
    elseif force_keyword == "i" # use interaction force
        F = f
    elseif force_keyword == "a"  # active force
        F = Ftot .- f
    else
        @warn "Received unsupported force keyword. Using total force for current modes."
        F = Ftot
    end

    if Ndim == 3
        @batch per=thread for t = 1:N_timesteps
            @turbo for i_k = 1:Nk
                kx = k_array[1, i_k]
                ky = k_array[2, i_k]
                kz = k_array[3, i_k]
                kmag = klengths[i_k]

                Rejkt = 0.0
                Imjkt = 0.0
                
                for particle = 1:N 
                    rx = r[1, particle, t]
                    ry = r[2, particle, t]
                    rz = r[3, particle, t]
                    fx = F[1, particle, t]
                    fy = F[2, particle, t]
                    fz = F[3, particle, t]

                    kr = kx*rx + ky*ry + kz*rz
                    kf = kx*fx + ky*fy + kz*fz

                    sinkr, coskr = sincos(kr)
                    Rejkt += kf * coskr / kmag
                    Imjkt += kf * sinkr / kmag
                end
                Rej[t, i_k] = Rejkt
                Imj[t, i_k] = Imjkt
            end
        end
    elseif Ndim == 2
        @batch per=thread for t = 1:N_timesteps
            @turbo for i_k = 1:Nk
                kx = k_array[1, i_k]
                ky = k_array[2, i_k]
                kmag = klengths[i_k]

                Rejkt = 0.0
                Imjkt = 0.0

                for particle = 1:N 
                    rx = r[1, particle, t]
                    ry = r[2, particle, t]
                    fx = F[1, particle, t]
                    fy = F[2, particle, t]

                    kr = kx*rx + ky*ry
                    kf = kx*fx + ky*fy

                    sinkr, coskr = sincos(kr)
                    Rejkt += kf * coskr / kmag
                    Imjkt += kf * sinkr / kmag
                end
                Rej[t, i_k] = Rejkt
                Imj[t, i_k] = Imjkt
            end
        end
    else
        throw(ArgumentError("Only 2D and 3D simulations are supported"))
    end
end



function find_current_modes_interaction(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace; verbose=true)
    Ndim, N, N_timesteps = size(s.r_array)
    Nk = kspace.Nk

    Rej = zeros(N_timesteps, Nk)
    Imj = zeros(N_timesteps, Nk)

    if verbose
        println("Calculating current modes for $N particles at $N_timesteps time points for $Nk wave vectors")
        println("Memory usage: $(Base.format_bytes(2*Base.summarysize(Rej)))")
        println("Based on 10 GFLOPS, this will take approximately $(round(Nk*s.N*N_timesteps*9/10^10, digits=1)) seconds.")
    end
    tstart = time()

    Ftot = calculate_total_force(s)
    _find_current_modes_partial!(Rej, Imj, s.r_array, Ftot, s.mobility .* s.F_array, kspace, force_keyword="i")

    if verbose
        tstop = time()
        println("Elapsed time: $(round(tstop-tstart,digits=3)) seconds")
        println("Achieved GFLOPS: $(round(Nk*s.N*N_timesteps*9/(tstop-tstart)/10^9, digits=3))")
    end

    return SingleComponentCurrentModesInt(Rej, Imj)
end

function find_current_modes_active(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace; verbose=true)
    Ndim, N, N_timesteps = size(s.r_array)
    Nk = kspace.Nk

    Rej = zeros(N_timesteps, Nk)
    Imj = zeros(N_timesteps, Nk)

    if verbose
        println("Calculating current modes for $N particles at $N_timesteps time points for $Nk wave vectors")
        println("Memory usage: $(Base.format_bytes(2*Base.summarysize(Rej)))")
        println("Based on 10 GFLOPS, this will take approximately $(round(Nk*s.N*N_timesteps*9/10^10, digits=1)) seconds.")
    end
    tstart = time()

    Ftot = calculate_total_force(s)
    _find_current_modes_partial!(Rej, Imj, s.r_array, Ftot, s.mobility .* s.F_array, kspace, force_keyword="a")

    if verbose
        tstop = time()
        println("Elapsed time: $(round(tstop-tstart,digits=3)) seconds")
        println("Achieved GFLOPS: $(round(Nk*s.N*N_timesteps*9/(tstop-tstart)/10^9, digits=3))")
    end

    return SingleComponentCurrentModesAct(Rej, Imj)
end

# UPDATE THESE LATER!! (Don't need them yet)
# function find_current_modes_interaction(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace; verbose=true)
#     N_species = length(s.r_array)
#     Nk = kspace.Nk
#     N_timesteps = size(s.r_array[1], 3)
#     klengths = kspace.k_lengths

#     Rej = [zeros(N_timesteps, Nk) for i = 1:N_species]
#     Imj = [zeros(N_timesteps, Nk) for i = 1:N_species]
#     if verbose
#         println("Calculating density modes for $(s.N) particles at $N_timesteps time points for $Nk wave vectors")
#         println("Memory usage: $(Base.format_bytes(2*Base.summarysize(Rej)))")
#         println("Based on 10 GFLOPS, this will take approximately $(round(Nk*s.N*N_timesteps*9/10^10, digits=1)) seconds.")
#     end
#     tstart = time()

#     for species in 1:N_species
#         _find_current_modes_partial!(Rej[species], Imj[species], s.r_array[species], s.F_array[species], s.u_array[species], s.v0[species], s.mobility[species], kspace, "i")
#     end

#     if verbose
#         tstop = time()
#         println("Elapsed time: $(round(tstop-tstart,digits=3)) seconds")
#         println("Achieved GFLOPS: $(round(Nk*s.N*N_timesteps*9/(tstop-tstart)/10^9, digits=3))")
#     end

#     return MultiComponentCurrentModesInt(Rej, Imj)
# end

# function find_current_modes_active(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace; verbose=true)
#     N_species = length(s.r_array)
#     Nk = kspace.Nk
#     N_timesteps = size(s.r_array[1], 3)
#     klengths = kspace.k_lengths

#     Rej = [zeros(N_timesteps, Nk) for i = 1:N_species]
#     Imj = [zeros(N_timesteps, Nk) for i = 1:N_species]
#     if verbose
#         println("Calculating density modes for $(s.N) particles at $N_timesteps time points for $Nk wave vectors")
#         println("Memory usage: $(Base.format_bytes(2*Base.summarysize(Rej)))")
#         println("Based on 10 GFLOPS, this will take approximately $(round(Nk*s.N*N_timesteps*9/10^10, digits=1)) seconds.")
#     end
#     tstart = time()

#     for species in 1:N_species
#         _find_current_modes_partial!(Rej[species], Imj[species], s.r_array[species], s.F_array[species], s.u_array[species], s.v0[species], s.mobility[species], kspace, "a")
#     end

#     if verbose
#         tstop = time()
#         println("Elapsed time: $(round(tstop-tstart,digits=3)) seconds")
#         println("Achieved GFLOPS: $(round(Nk*s.N*N_timesteps*9/(tstop-tstart)/10^9, digits=3))")
#     end

#     return MultiComponentCurrentModesAct(Rej, Imj)
# end


# function show(io::IO,  ::MIME"text/plain", ρkt::SingleComponentCurrentModes)
#     println(io, "SingleComponentCurrentModes with real and imaginary parts of size $(size(ρkt.Re)).")
# end

# function show(io::IO,  ::MIME"text/plain", ρkt::MultiComponentCurrentModes)
#     println(io, "MultiComponentCurrentModes with real and imaginary parts of size $(size(ρkt.Re[1])) for $(length(ρkt.Re)) species.")
# end


###############################################################################################################
# PARTIAL VELOCITY CORRELATIONS
##########################################################

# (cross-)correlation between two AbstractDensityMode objects -- for example, current & density modes, or partial velocity correlations
function find_static_cross_correlations(s::Simulation, kspace::KSpace, jkt1::AbstractDensityModes, jkt2::AbstractDensityModes, k_sample_array::AbstractVector; k_binwidth=0.1)
    wk_array = []
    for (ik, k) in enumerate(k_sample_array)
        kmin = k - k_binwidth/2
        kmax = k + k_binwidth/2
        push!(wk_array, find_static_cross_correlations(s, kspace, jkt1, jkt2; kmin=kmin, kmax=kmax))
    end
    return wk_array
end

function find_static_cross_correlations(s::Union{SingleComponentSimulation, SelfPropelledVoronoiSimulation}, kspace::KSpace, jkt1::AbstractDensityModes, jkt2::AbstractDensityModes; kmin=0.0, kmax=10.0^10.0)
    wk = real_static_correlation_function(jkt1.Re, jkt1.Im, jkt2.Re, jkt2.Im, kspace, kmin, kmax)
    return wk / s.N
end

function find_static_cross_correlations(s::Union{MultiComponentSimulation,MCSPVSimulation}, kspace::KSpace, jkt1::AbstractDensityModes, jkt2::AbstractDensityModes; kmin=0.0, kmax=10.0^10.0)
    N_species = s.N_species
    wk = zeros(N_species, N_species)
    for α=1:N_species
        for β = 1:N_species
            wk[α,β] = real_static_correlation_function(jkt1.Re[α], jkt1.Im[α], jkt2.Re[β], jkt2.Im[β], kspace, kmin, kmax)
        end
    end
    return wk / s.N
end
