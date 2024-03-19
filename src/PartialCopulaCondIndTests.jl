module PartialCopulaCondIndTests

import StatsAPI
using StatsAPI: HypothesisTest, nobs, pvalue


# Reference to the main paper that describes the methods implemented in this module
# This string is used in the documentation of the methods
paper = """[Petersen, Lasse, and Niels Richard Hansen. 
"Testing conditional independence via quantile regression based partial copulas."
Journal of Machine Learning Research 22.70 (2021): 1-47.]
(https://www.jmlr.org/papers/v22/20-1074.html)"""


abstract type IndependenceTest <: HypothesisTest end
abstract type ConditionalIndependenceTest <: HypothesisTest end
abstract type PartialCopulaEstimator end

struct PartialCopulaCondIndTest <: ConditionalIndependenceTest
    partial_copula_estimator::PartialCopulaEstimator
    independence_test::IndependenceTest
end


include("generalized_correlation.jl")

end
