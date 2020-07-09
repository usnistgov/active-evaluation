import pytest
import scipy.stats
import numpy as np
import aegis.acteval.samplers
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.system
import aegis.acteval.data_processor
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestConfidenceIntervals(object):
    """
    Class to test confidence interval computations.
    """

    def test_first_confidence_intervals(self):
        """
        A first test for simple confidence intervals.

        Returns: Nothing

        """
        # First, get the proper files that we need
        num_success_rounds_required = 1
        my_system = aegis.acteval.system.System("s1", [], [])
        my_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            aegis.acteval.strata.StrataFirstSystem(1, [my_system]), num_success_rounds_required
        )
        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric()
        p_hat = 0.5
        alpha = 0.05
        delta = 0.01
        var = 0.5
        n = 100
        expected_confidence_value = 0.13859038243496777
        my_system.sampled_trials = n
        my_system.score = p_hat
        my_system.score_variance = var/n
        conf_intervals = my_sampler.strata.get_confidence_intervals_all_systems(my_metric, alpha)
        obtained_confidence_value = conf_intervals[0][2]
        meets_criteria = my_sampler.meets_confidence_delta(obtained_confidence_value, delta)
        assert not meets_criteria
        assert conf_intervals[0] == [p_hat - expected_confidence_value,
                                     p_hat + expected_confidence_value,
                                     pytest.approx(expected_confidence_value, rel=1e-08)]

    def test_confidence_intervals(self):
        """
        A function that tests a variety of confidence intervals by looping over parameters.

        Returns: Nothing.

        """
        # First, get the proper files that we need
        # Assumes confidence interval is computed according to the normal distribution
        # And assumes that p_hat is irrelevant
        num_success_rounds_required = 1
        my_system = aegis.acteval.system.System("s1", [], [])
        my_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            aegis.acteval.strata.StrataFirstSystem(1, [my_system]), num_success_rounds_required
        )
        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric()
        n = 100
        for p_hat in np.linspace(0.1, 0.9, 9):
            for alpha in np.linspace(0.01, 0.1, 10):
                for delta in np.linspace(0.01, 0.1, 10):
                    for var in np.linspace(0.1, 0.5, 5):
                        expected_confidence_value = scipy.stats.norm.ppf(
                            1 - (alpha / 2), loc=0, scale=1
                        ) * np.sqrt(var/n)
                        my_system.sampled_trials = n
                        my_system.score = p_hat
                        my_system.score_variance = var / n
                        conf_intervals = my_sampler.strata.get_confidence_intervals_all_systems(
                            my_metric, alpha)
                        obtained_confidence_value = conf_intervals[0][2]
                        meets_criteria = my_sampler.meets_confidence_delta(
                            obtained_confidence_value, delta
                        )
                        expected_meets_criteria = expected_confidence_value <= delta
                        assert conf_intervals[0] == [p_hat - expected_confidence_value,
                                                     p_hat + expected_confidence_value,
                                                     pytest.approx(expected_confidence_value,
                                                                   rel=1e-08)], (
                                "failed with p_hat: "
                                + str(p_hat)
                                + " alpha: "
                                + str(alpha)
                                + " delta: "
                                + str(delta)
                                + " var: "
                                + str(var)
                        )
                        assert obtained_confidence_value == pytest.approx(
                            expected_confidence_value, rel=1e-08
                        ), (
                                "failed with p_hat: "
                                + str(p_hat)
                                + " alpha: "
                                + str(alpha)
                                + " delta: "
                                + str(delta)
                                + " var: "
                                + str(var)
                        )
                        assert meets_criteria == expected_meets_criteria, (
                            "failed with p_hat: "
                            + str(p_hat)
                            + " alpha: "
                            + str(alpha)
                            + " delta: "
                            + str(delta)
                            + " var: "
                            + str(var)
                        )
