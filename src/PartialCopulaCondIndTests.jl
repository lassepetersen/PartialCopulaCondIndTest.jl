module PartialCopulaCondIndTests

import StatsAPI
using StatsAPI: HypothesisTest, nobs, pvalue


abstract type IndependenceTest <: HypothesisTest end
abstract type ConditionalIndependenceTest <: HypothesisTest end
abstract type PartialCopulaEstimator end

struct PartialCopulaCondIndTest <: ConditionalIndependenceTest
    partial_copula_estimator::PartialCopulaEstimator
    independence_test::IndependenceTest
end


include("generalized_correlation.jl")

end
