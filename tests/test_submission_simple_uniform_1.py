import numpy as np
import pandas as pd
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestSubmissionSimpleUniform1(object):
    """
    Pytest class that run through different steps, using a known case as a focal point.
    Each method will use one submission to test different features and evaluate.
    """

    def test_single_system_uniform(self):
        """
        A case of 10000 samples with a system that takes the key and outputs a random uniform
        score based on the key. Key is 0 or 1.

        Returns: None

        """
        desired_seed = 2718
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/simple_uniform_1"
        key_fpath = "data/test/simple_uniform_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s_uniform"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        assert system_fpaths == [
            "data/test/simple_uniform_1/s_uniform_outputs.csv",
        ]

        assert threshold_fpaths == [
            "data/test/simple_uniform_1/s_uniform_thresholds.csv",
        ]
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        assert len(system_list) == 1
        # The number of ideal strata is 3, so we will use that case
        num_strata = 3
        my_strata = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata.stratify()
        assert my_strata.key_df.shape == (10002, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (10002, 3)
        # Check modification of combined_df
        assert combined_df.shape == (10002, 4)
        assert my_strata.num_trials == 10002
        # Check that we have four strata with the right index numbers
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
            1,
            2,
        ]
        assert my_strata.num_strata == 3
        for ind in range(0, len(my_strata.strata)):
            curr_strata = my_strata.strata[ind]
            strata_df = combined_df.loc[
                combined_df["stratum_index"] == ind, ["trial_id", "stratum_index", "key"]
            ]
            assert curr_strata.stratum_key_df.shape == strata_df.shape
        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric(key_values=[-1, 1])
        # Add decision conversion only once, so that it appears in all data frames
        [my_metric.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        assert my_strata.key_df.shape == (10002, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert combined_df.shape == (10002, 5)
        my_strata.add_samples_to_strata(init_df)
        assert my_strata.key_df.shape == (10002, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (10002, 3)
        assert combined_df.shape == (10002, 5)
        # Test initial samples
        # First, check that the total number of non missing keys is the same as the number of
        # trials in the initial data frame, which is 72
        strata_key_df = combined_df.loc[
            pd.notna(combined_df["key"]), :
        ]
        assert strata_key_df.shape[0] == 101
        assert my_strata.num_trials_sampled == 101
        for ind in range(0, len(my_strata.strata)):
            curr_strata = my_strata.strata[ind]
            init_stratum_df = strata_key_df.loc[
                strata_key_df["stratum_index"] == ind, :
            ]
            stratum_combined_df = curr_strata.get_combined_systems_df()
            curr_strata_key_df = stratum_combined_df.loc[
                pd.notna(stratum_combined_df["key"]), :
            ]
            assert curr_strata_key_df.shape[0] == init_stratum_df.shape[0]

        sys_samples = my_strata.estimate_samples_all_systems(my_metric)
        sys_pop = my_strata.estimate_pop_all_systems(my_metric)
        sys_score = my_strata.estimate_score_all_systems(my_metric)
        sys_score_var = my_strata.estimate_score_variance_all_systems(my_metric)
        assert sys_samples[0] == 101
        assert sys_pop[0] == 10002
        assert sys_score == [0.9166368022502356]
        assert sys_score_var == [0.0008664544411216739]

        # Now perform a round of sampling
        num_success_rounds_required = 1
        num_step_samples = 100
        adaptive_trial_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            my_strata, num_success_rounds_required
        )

        my_alpha = 0.10
        my_delta = 0.20

        my_strata.get_confidence_intervals_all_systems(my_metric, my_alpha)
        system_conf_values = [sys_conf[2] for sys_conf in my_strata.system_confidence_list]
        assert system_conf_values == [0.04841725045320777]
        my_strata.get_confidence_intervals_all_systems(my_metric, my_alpha)
        succ_conf_value = my_strata.aggregate_system_confidence_values()
        succ_round = adaptive_trial_sampler.meets_confidence_criteria(
            my_strata, my_delta, my_alpha, my_metric
        )
        assert succ_conf_value == 0.04841725045320777
        assert succ_round

        samples = adaptive_trial_sampler.draw_samples(
            num_step_samples, my_metric, rng
        )

        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (100, 2)

        my_strata.add_samples_to_strata(annotations_df)
        sys_samples = my_strata.estimate_samples_all_systems(my_metric)
        sys_pop = my_strata.estimate_pop_all_systems(my_metric)
        sys_score = my_strata.estimate_score_all_systems(my_metric)
        sys_score_var = my_strata.estimate_score_variance_all_systems(my_metric)
        assert sys_samples[0] == 201
        assert sys_pop[0] == 10002
        # system_conf_values = [my_metric.get_sys_confidence_value(sys, my_alpha)
        #                      for sys in my_strata.system_list]
        # succ_conf_value = my_strata.aggregate_system_confidence_values()
        # succ_round = adaptive_trial_sampler.meets_confidence_criteria(
        #     my_strata, my_delta, my_alpha, my_metric
        # )
