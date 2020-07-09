import math
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


class TestSubmissionSaeTest1Components(object):
    """
    Class of tests specifically for data set data/test/sae_test_1
    """

    def test_single_stratification_bins(self):
        """
        Test the stratification bins for the system "s1" to check the widths and the proportions

        """
        # no seed needed
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["s1"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystem
        # Start with Proportional Stratification
        bin_style = "perc"
        my_strata = strata_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert len(set(strata_size_counts)) == 1

        strata_type = aegis.acteval.strata.StrataFirstSystem
        # No perform equal width stratification
        bin_style = "equal"
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        my_strata = strata_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same widths of the bins
        combined_df = my_strata.get_combined_systems_df()
        min_val_df = combined_df.groupby('stratum_index').min()
        max_val_df = combined_df.groupby('stratum_index').max()
        range_df = max_val_df['s1'] - min_val_df['s1']
        expected_widths = [0.7550462481484059, 0.7548007456781021,
                           0.754716091755302, 0.7547110219335698]
        assert list(range_df) == expected_widths
        assert my_strata.num_strata == num_strata

    def test_two_system_known_case_stratify_first_substep(self):
        """
        Known case with two systems: The first normally distributed around the key with a sd of
        0.3 and the second system a no-information system that always returns 0. Key is 0 or 1.

        Returns: Nothing

        """
        desired_seed = 5314
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        # # Test with 4 strata
        num_strata = 4
        my_strata = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata.stratify(bin_style="equal")
        assert my_strata.key_df.shape == (10000, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (10000, 3)
        # Check modification of combined_df
        assert combined_df.shape == (10000, 5)
        assert my_strata.num_trials == 10000
        # Check that we have four strata with the right index numbers
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
            1,
            2,
            3,
        ]
        assert my_strata.num_strata == 4
        for ind in range(0, len(my_strata.strata)):
            curr_strata = my_strata.strata[ind]
            strata_df = combined_df.loc[
                combined_df["stratum_index"] == ind, ["trial_id", "stratum_index", "key"]
            ]
            assert curr_strata.stratum_key_df.shape == strata_df.shape
        my_strata.add_samples_to_strata(init_df)
        assert my_strata.key_df.shape == (10000, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (10000, 3)
        assert combined_df.shape == (10000, 5)
        # Test initial samples
        # First, check that the total number of non missing keys is the same as the number of
        # trials in the initial data frame, which is 72
        strata_key_df = combined_df.loc[
            pd.notna(combined_df["key"]), :
        ]
        assert strata_key_df.shape[0] == 72
        assert my_strata.num_trials_sampled == 72
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

        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric()
        [my_metric.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        sys_samples = my_strata.estimate_samples_all_systems(my_metric)
        sys_pop = my_strata.estimate_pop_all_systems(my_metric)
        sys_score = my_strata.estimate_score_all_systems(my_metric)
        sys_score_var = my_strata.estimate_score_variance_all_systems(my_metric)
        assert sys_samples == [72, 72]
        assert sys_pop[0] == 10000
        assert sys_pop[1] == 10000
        assert sys_score == [0.9868415213681034, 0.48270645007706053]
        assert sys_score_var == [0.0008114441448005382, 0.0008114441448005383]

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
        assert system_conf_values == [0.046855067968461506, 0.046855067968461506]
        my_strata.get_confidence_intervals_all_systems(my_metric, my_alpha)
        succ_conf_value = my_strata.aggregate_system_confidence_values()
        succ_round = adaptive_trial_sampler.meets_confidence_criteria(
            my_strata, my_delta, my_alpha, my_metric
        )
        assert succ_conf_value == pytest.approx(0.0468550679684615)
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
        assert sys_samples == [172, 172]
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (172, 7)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [25, 59, 53, 35]
        strata_samples_list = [
            stratum.estimate_samples_all_systems(my_metric)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(my_metric)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(my_metric)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(my_metric)
            for stratum in my_strata.strata
        ]
        # Given the strata lists, this hand checks that sys_pop,
        # sys_score, and sys_score_var are computed correctly
        assert strata_samples_list == [[25, 25], [59, 59], [53, 53], [35, 35]]
        assert strata_pop_list == [[826, 826], [4031, 4031], [4017, 4017], [1126, 1126]]
        assert strata_score_list == [[0.9923515425821422, 0.9923515425821422],
                                     [0.9978345845275225, 0.9978345845275225],
                                     [0.9223673431967632, 0.07763265680323693],
                                     [0.9953599670662807, 0.004640032933719115]]
        assert strata_score_var_list == [[0.0021711001532104266, 0.0021711001532104266],
                                         [0.0003314705004329218, 0.0003314705004329218],
                                         [0.0015596882914967988, 0.001559688291496802],
                                         [0.0010708324682186568, 0.001070832468218652]]
        assert sys_pop == [10000, 10000]
        assert sys_score == [0.9667878524941322, 0.5159028643865262]
        # Since internal computations can change, use previous values for now
        expected_sys_score_var = [0.00032905088897233146, 0.00032905088897233195]
        assert sys_score_var == expected_sys_score_var

    def test_single_stratification(self):
        """
        Test the stratification bins for the system "s1" to check the widths and the proportions

        """
        desired_seed = 5314
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["s1"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystem
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_success_rounds_required = 3
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10
        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        sampler_type = aegis.acteval.samplers.AdaptiveTrialSampler
        # Start with Proportional Stratification
        bin_style = "perc"
        my_strata = strata_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert len(set(strata_size_counts)) == 1

        [metric_obj.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        num_total_samples = init_df.shape[0]
        # In Initialization, Set first samples but do not do a successful round check
        my_strata.add_samples_to_strata(init_df)

        trial_sampler = sampler_type(my_strata, num_success_rounds_required)
        # One round of sampling
        samples = trial_sampler.draw_samples(
            num_step_samples, metric_obj, rng
        )
        num_total_samples += len(samples)
        annotations_df = oracle_ref.get_annotations(samples)
        my_strata.add_samples_to_strata(annotations_df)
        # Update stratum and system information; return values are not used and
        # thus discarded
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_obj
        )
        assert my_strata.num_strata == num_strata
        assert succ_round

    def test_strata_first_with_decision(self):
        desired_seed = 1289438347
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["s1"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10
        request_initial_samples = True
        initial_samples_requested = 200
        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        sampler_type = aegis.acteval.samplers.AdaptiveTrialSampler
        # Start with Proportional Stratification
        bin_style = "perc"

        [metric_obj.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = strata_type(num_strata, system_list)
        my_strata.dirty_strata_cache()
        assert my_strata.num_strata == 4

        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == num_strata

        combined_df = my_strata.get_combined_systems_df()

        # Since bin style is perc, check that each bin within the same decision has the
        # same size
        for ind in range(0, len(my_strata.strata)):
            curr_stratum = my_strata.strata[ind]
            assert curr_stratum.stratum_index == ind
            stratum_combined_df = curr_stratum.get_combined_systems_df()
            decision_values = stratum_combined_df['s1_dec'].unique().tolist()
            assert len(decision_values) == 1

        all_decision_values = combined_df['s1_dec'].unique().tolist()
        assert all_decision_values == [0, 1]
        for dec in all_decision_values:
            dec_df = combined_df.loc[combined_df['s1_dec'] == dec, :]
            strata_size_counts = dec_df['stratum_index'].value_counts().to_list()
            assert len(strata_size_counts) == 2
            # We have an odd number of values for each decision, so our stratum sizes
            # within the decisions differ by 1
            assert math.fabs(strata_size_counts[0] - strata_size_counts[1]) <= 1

        # Check that scores are in line with the decisions
        dec = 0
        dec_df = combined_df.loc[combined_df['s1_dec'] == dec, :]
        dec_scores = dec_df['s1'].unique()
        assert all([x <= 0.5 for x in dec_scores])
        dec = 1
        dec_df = combined_df.loc[combined_df['s1_dec'] == dec, :]
        dec_scores = dec_df['s1'].unique()
        assert all([x > 0.5 for x in dec_scores])

        # Get initial samples
        num_total_samples = 0
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_obj,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)
            num_total_samples += len(initial_samples)
        # No first samples, so provide initial samples

        assert num_total_samples == 200

        trial_sampler = sampler_type(my_strata, num_success_rounds_required)
        # One round of sampling
        samples = trial_sampler.draw_samples(
            num_step_samples, metric_obj, rng
        )
        num_total_samples += len(samples)
        annotations_df = oracle_ref.get_annotations(samples)
        my_strata.add_samples_to_strata(annotations_df)
        # Update stratum and system information; return values are not used and
        # thus discarded
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_obj
        )
        assert my_strata.num_strata == num_strata
        assert num_total_samples == 300
        assert succ_round

    def test_multi_strata_intersect_with_decision(self):
        desired_seed = 1289438347
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        num_strata = 16
        strata_type = aegis.acteval.strata.StrataMultiSystemIntersectDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10
        request_initial_samples = True
        initial_samples_requested = 200
        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        sampler_type = aegis.acteval.samplers.AdaptiveTrialSampler
        # Start with Proportional Stratification
        bin_style = "perc"

        [metric_obj.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = strata_type(num_strata, system_list)
        my_strata.dirty_strata_cache()
        assert my_strata.num_strata == 16

        my_strata.stratify(bin_style=bin_style)
        # s2 is a no information system
        assert my_strata.num_strata == 4

        combined_df = my_strata.get_combined_systems_df()

        # Since bin style is perc, check that each bin within the same decision has the
        # same size
        for ind in range(0, len(my_strata.strata)):
            curr_stratum = my_strata.strata[ind]
            assert curr_stratum.stratum_index == ind
            stratum_combined_df = curr_stratum.get_combined_systems_df()
            decision_values = stratum_combined_df['s1_dec'].unique().tolist()
            assert len(decision_values) == 1

        all_decision_values = combined_df['s1_dec'].unique().tolist()
        assert all_decision_values == [0, 1]
        for dec in all_decision_values:
            dec_df = combined_df.loc[combined_df['s1_dec'] == dec, :]
            strata_size_counts = dec_df['stratum_index'].value_counts().to_list()
            assert len(strata_size_counts) == 2
            # We have an odd number of values for each decision, so our stratum sizes
            # within the decisions differ by 1
            assert math.fabs(strata_size_counts[0] - strata_size_counts[1]) <= 1

        # Get initial samples
        num_total_samples = 0
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_obj,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)
            num_total_samples += len(initial_samples)
        # No first samples, so provide initial samples

        assert num_total_samples == 200

        trial_sampler = sampler_type(my_strata, num_success_rounds_required)
        # One round of sampling
        samples = trial_sampler.draw_samples(
            num_step_samples, metric_obj, rng
        )
        num_total_samples += len(samples)
        annotations_df = oracle_ref.get_annotations(samples)
        my_strata.add_samples_to_strata(annotations_df)
        # Update stratum and system information; return values are not used and
        # thus discarded
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_obj
        )
        assert my_strata.num_strata == 4
        assert num_total_samples == 300
        assert succ_round


class TestSubmissionSaeTest1ExperimentRuns(object):
    """
    Class of tests specifically for data set data/test/sae_test_1
    """

    def test_simple_controller_call(self):
        """
        First test of complete controller execution. Stops at first successful round
        Known case with two systems: The first normally distributed around the key with a sd of
        0.3 and the second system a no-information system that always returns 0. Key is 0 or 1.

        Returns: Nothing

        """
        desired_seed = 5314
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystem
        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_success_rounds_required = 2
        num_step_samples = 100
        my_alpha = 0.10
        my_delta = 0.20
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        # Uses default number of initial samples, so test will fail when the default
        # value of number of initial samples changes
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=my_alpha, delta=my_delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=my_metric,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="equal")
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng)
        assert len(my_report.system_list) == 2
        # Checks that the default value has not changed for
        # initial_samples_per_bin
        assert my_report.experiment.initial_samples == 200
        assert my_report.system_list[0].system_id == "s1"
        assert my_report.system_list[1].system_id == "s2"
        assert my_report.system_list[0].sampled_trials == 400
        assert my_report.system_list[1].sampled_trials == 400
        assert my_report.system_list[0].population == 10000
        assert my_report.system_list[1].population == 10000
        assert my_report.system_list[0].score == 0.977466555160188
        assert my_report.system_list[1].score == 0.4874796768013349
        assert my_report.system_list[0].score_variance == 8.076274844293206e-05
        assert my_report.system_list[1].score_variance == 9.05617159438505e-05
        assert my_report.system_list[0].confidence_value == 0.015236216574723627
        assert my_report.system_list[1].confidence_value == 0.01613406913970411
        assert my_report.num_requested_init_trials == 128
        assert my_report.num_rounds == 2

        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=my_alpha, delta=my_delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=my_metric,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="equal", request_initial_samples=True,
                             initial_samples=100)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng)
        assert len(my_report.system_list) == 2
        assert my_report.system_list[0].system_id == "s1"
        assert my_report.system_list[1].system_id == "s2"
        assert my_report.system_list[0].sampled_trials == 304
        assert my_report.system_list[1].sampled_trials == 304
        assert my_report.system_list[0].population == 10000
        assert my_report.system_list[1].population == 10000
        assert my_report.system_list[0].score == 0.9670599521130216
        assert my_report.system_list[1].score == 0.5036685915928018
        assert my_report.system_list[0].score_variance == 0.0001485622105380214
        assert my_report.system_list[1].score_variance == 0.0001840737475487679
        assert my_report.system_list[0].confidence_value == 0.020664541263256897
        assert my_report.system_list[1].confidence_value == 0.02300210170518041
        assert my_report.num_requested_init_trials == 32
        assert my_report.num_rounds == 2
        assert my_report.experiment.initial_samples == 100

    def test_simple_controller_call_no_init_df(self):
        """
        Test with no initial samples

        Returns: Nothing

        """
        desired_seed = 5314
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystem
        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_success_rounds_required = 2
        num_step_samples = 100
        my_alpha = 0.10
        my_delta = 0.20
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=my_alpha, delta=my_delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=my_metric,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="equal", request_initial_samples=True,
                             initial_samples=8)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng)
        assert len(my_report.system_list) == 2
        assert my_report.system_list[0].system_id == "s1"
        assert my_report.system_list[1].system_id == "s2"
        assert my_report.system_list[0].sampled_trials == 208
        assert my_report.system_list[1].sampled_trials == 208
        assert my_report.system_list[0].population == 10000
        assert my_report.system_list[1].population == 10000
        assert my_report.system_list[0].score == 0.9427729914537637
        assert my_report.system_list[1].score == 0.5083046947496014
        assert my_report.system_list[0].score_variance == 0.00032786877038167744
        assert my_report.system_list[1].score_variance == 0.00034320014225369265
        assert my_report.system_list[0].confidence_value == 0.030698815465144857
        assert my_report.system_list[1].confidence_value == 0.031408364465474126
        assert my_report.num_requested_init_trials == 8
        assert my_report.num_rounds == 2
        assert my_report.experiment.initial_samples == 8

        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=my_alpha, delta=my_delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=my_metric,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="equal", request_initial_samples=True,
                             initial_samples=40)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng)
        assert len(my_report.system_list) == 2
        assert my_report.system_list[0].system_id == "s1"
        assert my_report.system_list[1].system_id == "s2"
        assert my_report.system_list[0].sampled_trials == 240
        assert my_report.system_list[1].sampled_trials == 240
        assert my_report.system_list[0].population == 10000
        assert my_report.system_list[1].population == 10000
        assert my_report.system_list[0].score == 0.9598911773134821
        assert my_report.system_list[1].score == 0.5274432053793294
        assert my_report.system_list[0].score_variance == 0.0002142502890654511
        assert my_report.system_list[1].score_variance == 0.00026177756343755387
        assert my_report.system_list[0].confidence_value == 0.024816029718191968
        assert my_report.system_list[1].confidence_value == 0.02743075762740721
        assert my_report.num_requested_init_trials == 40
        assert my_report.num_rounds == 2
        assert my_report.experiment.initial_samples == 40

        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=my_alpha, delta=my_delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=my_metric,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="equal", request_initial_samples=True,
                             initial_samples=100)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng)
        assert len(my_report.system_list) == 2
        assert my_report.system_list[0].system_id == "s1"
        assert my_report.system_list[1].system_id == "s2"
        assert my_report.system_list[0].sampled_trials == 300
        assert my_report.system_list[1].sampled_trials == 300
        assert my_report.system_list[0].population == 10000
        assert my_report.system_list[1].population == 10000
        assert my_report.system_list[0].score == 0.9435470326861188
        assert my_report.system_list[1].score == 0.4958211540900709
        assert my_report.system_list[0].score_variance == 0.0002240921942529699
        assert my_report.system_list[1].score_variance == 0.0002316480835290025
        assert my_report.system_list[0].confidence_value == 0.025379610741914527
        assert my_report.system_list[1].confidence_value == 0.025803935524047972
        assert my_report.num_requested_init_trials == 100
        assert my_report.num_rounds == 2
        assert my_report.experiment.initial_samples == 100

    def test_simple_controller_call_three_successful_rounds(self):
        """
        Second test of complete controller execution. Stops after three consecutive successful
        rounds.
        Known case with two systems: The first normally distributed around the key with a sd of
        0.3 and the second system a no-information system that always returns 0. Key is 0 or 1.

        Returns: Nothing

        """
        desired_seed = 5314
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_1"
        key_fpath = "data/test/sae_test_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystem
        my_metric = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_success_rounds_required = 4
        num_step_samples = 100
        my_alpha = 0.10
        my_delta = 0.20
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples,
                             alpha=my_alpha, delta=my_delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=my_metric,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="equal", request_initial_samples=True,
                             initial_samples=200)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng)
        assert len(my_report.system_list) == 2
        assert my_report.experiment.initial_samples == 200
        assert my_report.system_list[0].system_id == "s1"
        assert my_report.system_list[1].system_id == "s2"
        assert my_report.system_list[0].sampled_trials == 600
        assert my_report.system_list[1].sampled_trials == 600
        assert my_report.system_list[0].population == 10000
        assert my_report.system_list[1].population == 10000
        assert my_report.system_list[0].score == 0.9612296432596188
        assert my_report.system_list[1].score == 0.4889552435608131
        assert my_report.system_list[0].score_variance == 7.164890515550521e-05
        assert my_report.system_list[1].score_variance == 8.145660118227595e-05
        assert my_report.system_list[0].confidence_value == 0.01435080857761073
        assert my_report.system_list[1].confidence_value == 0.015301525654868198
        assert my_report.num_rounds == 4
