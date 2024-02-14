using PartialCopulaCondIndTests
using Test

using QuadGK
using Distributions

@testset "trimming function" begin
    q = 10
    tau_sequence = range(0.01, stop=0.99, length=q + 1)

    for i in 1:q
        σ = PartialCopulaCondIndTests.TrimmingFunc(tau_sequence[i], tau_sequence[i + 1])
        @test isapprox(quadgk(u -> σ(u), tau_sequence[i], tau_sequence[i + 1])[1], 1.0, atol=1e-5)
    end
end


@testset "phi function" begin
    q = 10
    tau_sequence = range(0.01, stop=0.99, length=q + 1)

    for i in 1:q
        φ = PartialCopulaCondIndTests.PhiFunc(tau_sequence[i], tau_sequence[i + 1])
        @test isapprox(quadgk(u -> φ(u), tau_sequence[i], tau_sequence[i + 1])[1], 0.0, atol=1e-5)
        @test isapprox(quadgk(u -> φ(u)^2, tau_sequence[i], tau_sequence[i + 1])[1], 1.0, atol=1e-5)
    end
end


@testset "Trimmed Spearman correlation" begin
    n = 1000
    q = 10
    
    unifom = Uniform(0, 1)
    U₁ = rand(unifom, n)
    U₂ = rand(unifom, n)

    tau_sequence = range(0.01, stop=0.99, length=q + 1)
    
    ρ = PartialCopulaCondIndTests.TrimmedSpearmanCorrelation(q)
    ρ_hat = ρ(U₁, U₂)

    @test size(ρ_hat) == (q, q)
    @test !any(isnan, ρ_hat)
    @test all(ρ_hat .<= 1.0)
    @test all(ρ_hat .>= -1.0)
end
