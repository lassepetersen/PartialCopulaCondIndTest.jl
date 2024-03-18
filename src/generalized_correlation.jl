using QuadGK
using Distributions
using Statistics: mean
using LinearAlgebra: norm


struct TrimmingFunc
    # Trimming parameters
    μ::Float64
    λ::Float64
    
    # Linear approximation parameters
    δ::Float64

    # Lipschipz bump function approximation
    f::Function
end


function valid_trimming_params(μ, λ, δ)
    if !(0 <= μ <= 1)
        throw(ArgumentError("μ must be in [0, 1]"))
    end

    if !(0 <= λ <= 1)
        throw(ArgumentError("λ must be in [0, 1]"))
    end

    if !(μ < λ)
        throw(ArgumentError("μ must be less than λ"))
    end

    if !(0 <= δ <= (λ - μ) / 2)
        throw(ArgumentError("δ must be in [0, (λ - μ) / 2]"))
    end
end


function TrimmingFunc(μ, λ, δ)
    # Assert that the parameters are valid
    valid_trimming_params(μ, λ, δ)

    # Compute the normalization constant
    K = 1 / (λ - μ - δ)

    # Define the Lipschipz bump function approximation
    f = function(x)
        if μ + δ <= x <= λ - δ
            return K
        elseif (x < μ) | (x > λ)
            return 0
        elseif (μ <= x < μ + δ)
            return K * (x - μ) / δ
        elseif (λ - δ < x <= λ)
            return K * (λ - x) / δ
        end
    end

    return TrimmingFunc(μ, λ, δ, f)
end


TrimmingFunc(μ, λ) = TrimmingFunc(μ, λ, 0.1 * (λ - μ) / 2)


function (σ::TrimmingFunc)(u::Float64)
    return σ.f(u)
end


struct PhiFunc
    # Centering and scaling parameters
    m::Float64
    c::Float64
    
    # Trimming function
    σ::TrimmingFunc
end

function PhiFunc(μ, λ)
    σ = TrimmingFunc(μ, λ)

    m = quadgk(u -> u * σ(u), μ, λ)[1]

    c = 1 / sqrt(
        quadgk(u -> (u - m)^2 * σ(u)^2, μ, λ)[1]
    )

    if !isfinite(m)
        throw(ErrorException("m is not finite, δ could be too small"))
    end

    if !isfinite(c)
        throw(ErrorException("c is not finite, δ could be too small"))
    end

    return PhiFunc(m, c, σ)
end


function (φ::PhiFunc)(u::Float64)
    return φ.c * (u - φ.m) * φ.σ(u)
end


function valid_tau_params(τ_min, τ_max)
    if !(0 < τ_min < 1)
        throw(ArgumentError("τ_min must be in (0, 1)"))
    end

    if !(0 < τ_max < 1)
        throw(ArgumentError("τ_max must be in (0, 1)"))
    end

    if !(τ_min < τ_max)
        throw(ArgumentError("τ_min must be less than τ_max"))
    end
end


struct TrimmedSpearmanCorrelation
    # Dimension 
    q::Int

    # Tau parameters
    τ_min::Float64
    τ_max::Float64

    # Phi functions
    phi_vector::Vector{PhiFunc}
end


function TrimmedSpearmanCorrelation(q, τ_min=0.01, τ_max=0.99)
    # Assert that the parameters are valid
    valid_tau_params(τ_min, τ_max)

    # Initialize the Phi vector
    phi_vector = Vector{PhiFunc}(undef, q)

    tau_sequence = range(τ_min, stop=τ_max, length=q + 1)
    for i in 1:q
        phi_vector[i] = PhiFunc(tau_sequence[i], tau_sequence[i + 1])
    end

    return TrimmedSpearmanCorrelation(q, τ_min, τ_max, phi_vector)
end


function (ρ::TrimmedSpearmanCorrelation)(U₁::Vector{Float64}, U₂::Vector{Float64})
    return [mean(ρ.phi_vector[i].(U₁) .* ρ.phi_vector[j].(U₂)) for i in 1:ρ.q, j in 1:ρ.q]
end


struct TrimmedSpearmanCorrelationTest <: IndependenceTest
    q::Int
    U₁::Vector{Float64}
    U₂::Vector{Float64}

    function TrimmedSpearmanCorrelationTest(q, U₁, U₂)
        if length(U₁) == length(U₂)
            new(q, U₁, U₂)
        else
            throw(ArgumentError("Generalized residuals, U₁ and U₂, must have the same length"))    
        end
    end
end


testname(::TrimmedSpearmanCorrelationTest) = "Trimmed Spearman correlation independence test"


function StatsAPI.nobs(TSCT::TrimmedSpearmanCorrelationTest)
    return length(TSCT.U₁)
end


function StatsAPI.pvalue(TSCT::TrimmedSpearmanCorrelationTest)
    # Sample size
    n = nobs(TSCT)

    # Compute the trimmed Spearman correlation
    ρ = TrimmedSpearmanCorrelation(TSCT.q)
    ρ_hat = ρ(TSCT.U₁, TSCT.U₂)

    # Computing the test statistic
    T = norm(ρ_hat, 2)^2

    # Computing the p-value
    target_dist = Chisq(TSCT.q^2)
    p_value = ccdf(target_dist, n * T)

    return p_value
end
