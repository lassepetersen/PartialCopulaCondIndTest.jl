module PartialCopulaCondIndTests

import StatsAPI
using StatsAPI: HypothesisTest, nobs, pvalue


include("conditional_independence_test.jl")
include("generalized_correlation.jl")

end
