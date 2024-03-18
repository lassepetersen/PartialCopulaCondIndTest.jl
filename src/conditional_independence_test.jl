abstract type ConditionalIndependenceTest <: HypothesisTest end
abstract type IndependenceTest <: HypothesisTest end
abstract type PartialCopulaEstimator end


struct PartialCopulaCondIndTest <: ConditionalIndependenceTest
    partial_copula_estimator::PartialCopulaEstimator
    independence_test::IndependenceTest
end
