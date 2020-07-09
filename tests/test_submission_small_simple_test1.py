import numpy as np
import pandas as pd
import pytest
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestSubmissionSmallSimpleTest1(object):
    """
    Pytest class that run through different steps, using a known case as a focal point.
    Each method will use one submission to test different features and evaluate.
    """
    def test_small_simple_test1_case_one_strata(self):
        """
        A super-simple case of 20 trials with one system, thresholded at 0.
        First 12 are scored as -0.1 scoring 6 of them (50%) correctly,
        and the last 8 trials are scored at 0.1, 6 (75%) of them scored correctly.
        The entire key is provided initially, key is -1 or 1. Stratified in one strata only
        to map to uniform scoring.

        Returns: Nothing.

        """
        desired_seed = 29
        np.random.seed(seed=desired_seed)
        input_dir = "data/test/small_simple_test1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        assert system_fpaths == ["data/test/small_simple_test1/simple_outputs.csv"]
        assert threshold_fpaths == ["data/test/small_simple_test1/simple_thresholds.csv"]
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        assert len(system_list) == 1
        # This case uses only one strata
        num_strata = 1
        my_strata = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata.stratify()
        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (20, 3)
        # Check modification of combined_df
        assert combined_df.shape == (20, 4)
        assert my_strata.num_trials == 20
        # Check that we have four strata with the right index numbers
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
        ]
        assert my_strata.num_strata == 1
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
        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert combined_df.shape == (20, 5)
        my_strata.add_samples_to_strata(init_df)
        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (20, 3)
        assert combined_df.shape == (20, 5)
        strata_key_df = combined_df.loc[pd.notna(combined_df["key"]), :]
        assert strata_key_df.shape[0] == 20
        assert my_strata.num_trials_sampled == 20
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
        assert sys_pop[0] == 20
        assert sys_samples[0] == 20
        # This score and variance were calculated by hand
        # should be 0.6 if the bias has a finite population correction
        assert sys_score == [0.6]
        assert sys_score_var == [0]

        num_success_rounds_required = 1
        # num_step_samples = 100
        adaptive_trial_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            my_strata, num_success_rounds_required
        )

        # First parameters
        my_alpha = 0.10
        my_delta = 0.20

        my_strata.get_confidence_intervals_all_systems(my_metric, my_alpha)
        system_conf_values = [sys_conf[2] for sys_conf in my_strata.system_confidence_list]
        assert system_conf_values == [0.0]
        succ_round = adaptive_trial_sampler.meets_confidence_criteria(
            my_strata, my_delta, my_alpha, my_metric
        )
        succ_conf_value = my_strata.aggregate_system_confidence_values()
        assert succ_conf_value == 0.0
        assert succ_round

        # Use tighter parameters
        my_alpha = 0.05
        my_delta = 0.10

        my_strata.get_confidence_intervals_all_systems(my_metric, my_alpha)
        system_conf_values = [sys_conf[2] for sys_conf in my_strata.system_confidence_list]
        assert system_conf_values == [0.0]
        succ_conf_value = my_strata.aggregate_system_confidence_values()
        assert succ_conf_value == 0.0
        succ_round = adaptive_trial_sampler.meets_confidence_criteria(
            my_strata, my_delta, my_alpha, my_metric
        )
        assert succ_round

        # Next, sample and check that we cannot get any more samples
        pass

    def test_small_simple_test1_case_two_strata(self):
        """
        A super-simple case of 20 trials with one system, thresholded at 0.
        First 12 are scored as -0.1 scoring 6 of them (50%) correctly,
        and the last 8 trials are scored at 0.1, 6 (75%) of them scored correctly.
        The entire key is provided initially, key is -1 or 1. Stratified in two strata,
        equal-width bin so the stratification places all scores of -0.1 in one strata and
        scores of 0.1 in the other strata.

        Returns: Nothing.

        """
        input_dir = "data/test/small_simple_test1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        assert system_fpaths == ["data/test/small_simple_test1/simple_outputs.csv"]
        assert threshold_fpaths == ["data/test/small_simple_test1/simple_thresholds.csv"]
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        assert len(system_list) == 1
        # This case uses only one strata
        num_strata = 2
        my_strata = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata.stratify(bin_style='equal')
        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (20, 3)
        # Check that the widths of the bins are equal by checking that we have the proper
        # stratum sizes

        # Check modification of combined_df
        assert combined_df.shape == (20, 4)
        assert my_strata.num_trials == 20
        # Check that we have four strata with the right index numbers
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
            1
        ]
        assert my_strata.num_strata == 2
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
        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert combined_df.shape == (20, 5)
        my_strata.add_samples_to_strata(init_df)
        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (20, 3)
        assert combined_df.shape == (20, 5)
        strata_key_df = combined_df.loc[pd.notna(combined_df["key"]), :]
        assert strata_key_df.shape[0] == 20
        assert my_strata.num_trials_sampled == 20
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

        sys_pop = my_strata.estimate_pop_all_systems(my_metric)
        sys_score = my_strata.estimate_score_all_systems(my_metric)
        sys_score_var = my_strata.estimate_score_variance_all_systems(my_metric)
        assert sys_pop[0] == 20
        # This score and variance were calculated by hand
        # score computed to be 0.6
        assert sys_score == [pytest.approx(0.6)]
        assert sys_score_var == [0]
        # Next, sample and check that we cannot get any more samples
        pass

    def test_small_simple_test1_case_two_strata_get_init(self):
        """
        A super-simple case of 20 trials with one system, thresholded at 0.
        First 12 are scored as -0.1 scoring 6 of them (50%) correctly,
        and the last 8 trials are scored at 0.1, 6 (75%) of them scored correctly.
        The entire key is provided initially, key is -1 or 1. Stratified in two strata,
        equal-width bin so the stratification places all scores of -0.1 in one strata and
        scores of 0.1 in the other strata.

        Returns: Nothing.

        """
        desired_seed = 1412
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        # This case uses only 2 stratum
        num_strata = 2
        my_strata = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata.stratify(bin_style='equal')

        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric(key_values=[-1, 1])
        # Add decision conversion only once, so that it appears in all data frames
        [my_metric.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()

        initial_samples = my_strata.find_needed_initial_samples(my_metric,
                                                                initial_samples=4,
                                                                rng=rng)
        assert len(initial_samples) == 4
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(initial_samples)
        assert annotations_df.shape == (4, 2)

        my_strata.add_samples_to_strata(annotations_df)

        assert my_strata.key_df.shape == (20, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (20, 3)
        assert combined_df.shape == (20, 5)
        strata_key_df = combined_df.loc[pd.notna(combined_df["key"]), :]
        assert strata_key_df.shape[0] == 4
        assert my_strata.num_trials_sampled == 4
        for ind in range(0, len(my_strata.strata)):
            curr_strata = my_strata.strata[ind]
            stratum_combined_df = curr_strata.get_combined_systems_df()
            curr_strata_key_df = stratum_combined_df.loc[
                                 pd.notna(stratum_combined_df["key"]), :
                                 ]
            assert curr_strata_key_df.shape[0] == 2

        sys_pop = my_strata.estimate_pop_all_systems(my_metric)
        sys_score = my_strata.estimate_score_all_systems(my_metric)
        sys_score_var = my_strata.estimate_score_variance_all_systems(my_metric)
        assert sys_pop[0] == 20
        # This score and variance were calculated by hand
        # sys_score is 0.7 without bias
        assert sys_score == [pytest.approx(0.6245259044526774)]
        assert sys_score_var == [0.007886391354060242]
        # Next, sample and check that we cannot get any more samples
        pass
