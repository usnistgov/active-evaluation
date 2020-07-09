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


class TestSubmissionSaeTest2(object):
    """
    Class of tests specifically for data set data/test/sae_test_2
    """

    def test_random_sampling_4s(self):
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 80
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        my_strata = stratification_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [2500, 2500, 2500, 2500]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20, 20, 20, 20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [80]
        assert sys_pop[0] == 10000
        assert sys_score == [pytest.approx(0.9157737027167744)]
        assert sys_score_var == [0.0012631148640071647]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [17, 9, 12, 12]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [37, 29, 32, 32]

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [7, 11, 16, 16]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (180, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [44, 40, 48, 48]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_random_sampling_1s(self):
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 1
        request_initial_samples = True
        initial_samples_requested = 20
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        my_strata = stratification_type(num_strata, system_list)
        assert my_strata.num_strata == 1
        assert not my_strata.stratify_for_pure_random
        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [10000]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [20]
        assert sys_pop[0] == 10000
        assert sys_score == [pytest.approx(0.9401760668043981)]
        assert sys_score_var == [0.0046778369726087204]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (120, 5)
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_score == [pytest.approx(0.957645334752634)]
        assert sys_score_var == [0.0003740298243769716]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_random_sampling_n1s(self):
        """
        These numbers are the same as test_random_rampling_1s, so if there is an error with the
        numbers check if the previous test passes.

        Returns:

        """
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = -1
        request_initial_samples = True
        initial_samples_requested = 20
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]

        my_strata = stratification_type(num_strata, system_list)
        my_strata.dirty_strata_cache()
        assert my_strata.num_strata == 1
        assert my_strata.stratify_for_pure_random
        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [10000]

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [20]
        assert sys_pop[0] == 10000
        assert sys_score == [pytest.approx(0.9401760668043981)]
        assert sys_score_var == [0.0046778369726087204]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (120, 5)
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_score == [pytest.approx(0.957645334752634)]
        assert sys_score_var == [0.0003740298243769716]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_random_sampling_n1sd(self):
        """
        These numbers are the same as test_random_rampling_1s, so if there is an error with the
        numbers check if the previous test passes.

        Returns:

        """
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystemDecision
        alpha = 0.05
        num_strata = -1
        request_initial_samples = True
        initial_samples_requested = 20
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]

        my_strata = stratification_type(num_strata, system_list)
        my_strata.dirty_strata_cache()
        assert my_strata.num_strata == 1
        assert my_strata.stratify_for_pure_random
        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [10000]

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [20]
        assert sys_pop[0] == 10000
        assert sys_score == [pytest.approx(0.9401760668043981)]
        assert sys_score_var == [0.0046778369726087204]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (120, 5)
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_score == [pytest.approx(0.957645334752634)]
        assert sys_score_var == [0.0003740298243769716]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_random_sampling_1s_multi(self):
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1", "s2"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataMultiSystemIntersect
        alpha = 0.05
        num_strata = 1
        request_initial_samples = True
        initial_samples_requested = 20
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = stratification_type(num_strata, system_list)
        assert my_strata.num_strata == 1
        assert not my_strata.stratify_for_pure_random
        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [10000]

        my_strata.dirty_strata_cache()

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples[0] == [20]
        assert sys_pop[0] == 10000
        assert sys_score[0] == [pytest.approx(0.9401760668043981)]
        assert sys_score_var[0] == [0.0046778369726087204]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (120, 7)
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_score[0] == [pytest.approx(0.957645334752634)]
        assert sys_score_var[0] == [0.0003740298243769716]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert not succ_round

    def test_random_sampling_n1s_multi(self):
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1", "s2"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataMultiSystemIntersectDecision
        alpha = 0.05
        num_strata = -1
        request_initial_samples = True
        initial_samples_requested = 20
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = stratification_type(num_strata, system_list)
        assert my_strata.num_strata == 1
        assert my_strata.stratify_for_pure_random
        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [10000]

        my_strata.dirty_strata_cache()

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples[0] == [20]
        assert sys_pop[0] == 10000
        assert sys_score[0] == [pytest.approx(0.9401760668043981)]
        assert sys_score_var[0] == [0.0046778369726087204]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (120, 7)
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_score[0] == [pytest.approx(0.957645334752634)]
        assert sys_score_var[0] == [0.0003740298243769716]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert not succ_round

    def test_random_sampling_n1sd_multi(self):
        desired_seed = 1857673
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1", "s2"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataMultiSystemIntersectDecision
        alpha = 0.05
        num_strata = -1
        request_initial_samples = True
        initial_samples_requested = 20
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = stratification_type(num_strata, system_list)
        assert my_strata.num_strata == 1
        assert my_strata.stratify_for_pure_random
        my_strata.stratify(bin_style=bin_style)
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [10000]

        my_strata.dirty_strata_cache()

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [20]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples[0] == [20]
        assert sys_pop[0] == 10000
        assert sys_score[0] == [pytest.approx(0.9401760668043981)]
        assert sys_score_var[0] == [0.0046778369726087204]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Random
        trial_sampler = aegis.acteval.samplers.RandomTrialSampler(
            my_strata, num_success_rounds_required
        )

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        assert samples.shape[0] == 50
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)

        # Sampler 2: Random Fixed
        trial_sampler = aegis.acteval.samplers.RandomFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (120, 7)
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_score[0] == [pytest.approx(0.957645334752634)]
        assert sys_score_var[0] == [0.0003740298243769716]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert not succ_round

    def test_precision_sampling_1(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        my_strata = stratification_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [2500, 2500, 2500, 2500]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [0, 10, 10, 10]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [30]
        assert sys_pop[0] == 5039
        assert sys_score == [pytest.approx(0.9074958122472223)]
        assert sys_score_var == [0.00329458975485321]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Uniform
        trial_sampler = aegis.acteval.samplers.UniformFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        assert strata_samples_list == [[0], [10], [10], [10]]
        assert strata_pop_list == [[0], [39], [2500], [2500]]
        assert strata_score_list == [[np.nan],
                                     [0.6858190709046454],
                                     [0.8637554585152839],
                                     [0.954694323144105]]
        assert strata_score_var_list == [[np.nan],
                                         [0.018078512396694214],
                                         [0.00981404958677686],
                                         [0.003615702479338843]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 38
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [12, 13, 13]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (38, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (68, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [22, 23, 23]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] ==\
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert not succ_round

        # Sampler 2: Proportional Fixed
        trial_sampler = aegis.acteval.samplers.ProportionalFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [25, 25]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (118, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [22, 48, 48]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] ==\
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

        # Sampler 3: Adaptive Fixed
        trial_sampler = aegis.acteval.samplers.AdaptiveFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        assert strata_samples_list == [[0], [22], [48], [48]]
        assert strata_pop_list == [[0], [39], [2500], [2500]]
        assert strata_score_list == [[np.nan],
                                     [0.5445486518171161],
                                     [0.8674879905885429],
                                     [0.9899839874513905]]
        assert strata_score_var_list == [[np.nan],
                                         [0.010337901701323251],
                                         [0.002301124531445231],
                                         [0.00020199916701374427]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        assert samples_df.shape[0] == 38
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [12, 13, 13]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (38, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (156, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [34, 61, 61]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] ==\
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_recall_sampling_1(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryRecallMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        my_strata = stratification_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [2500, 2500, 2500, 2500]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [10, 10, 10, 10]

        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [19]
        assert sys_pop[0] == 4750
        assert sys_score == [pytest.approx(0.9525465465454414)]
        assert sys_score_var == [0.0019634255183043486]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Uniform
        trial_sampler = aegis.acteval.samplers.UniformFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        assert strata_samples_list == [[0], [0], [9], [10]]
        assert strata_pop_list == [[0], [0], [2250], [2500]]
        assert strata_score_list == [[np.nan],
                                     [np.nan],
                                     [0.9501601281024821],
                                     [0.954694323144105]]
        assert strata_score_var_list == [[np.nan],
                                         [np.nan],
                                         [0.004318181818181818],
                                         [0.003615702479338843]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [12, 12, 13, 13]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (90, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [22, 22, 23, 23]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        my_strata.estimate_score_variance_upper_all_systems(metric_object, alpha)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert not succ_round

        # Sampler 2: Proportional Fixed
        trial_sampler = aegis.acteval.samplers.ProportionalFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [1, 4, 22, 23]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (140, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [23, 26, 45, 46]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        my_strata.estimate_score_variance_upper_all_systems(metric_object, alpha)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

        # Sampler 3: Adaptive Fixed
        trial_sampler = aegis.acteval.samplers.AdaptiveFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_upper_list = [
            stratum.estimate_pop_upper_all_systems(metric_object, alpha)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_upper_list = [
            stratum.estimate_score_variance_upper_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        assert strata_samples_list == [[0], [2], [40], [46]]
        assert strata_pop_list == [[0], [192], [2222], [2500]]
        assert strata_pop_upper_list == [[139.96704278367065],
                                         [443.0859523893434],
                                         [2484.2184416515893],
                                         [2752.6935619445817]]
        assert strata_score_list == [[np.nan], [0.1660839160839161], [0.9880138867526531],
                                     [0.9895492641046607]]
        assert strata_score_var_list == [[np.nan],
                                         [0.034722222222222224],
                                         [0.00028681907028129517],
                                         [0.00021927342688999547]]
        assert strata_score_var_upper_list == [[0.24779911964785914],
                                               [0.034722222222222224],
                                               [0.00028681907028129517],
                                               [0.00021927342688999547]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [15, 17, 9, 9]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (190, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [38, 43, 54, 55]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape[0] == stratum_score_df.shape[0]
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape[0] == \
                stratum_combined_df.shape[0]
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_population_estimates_1(self):

        # ========================
        # # # Code to setup test
        # ========================
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object_r = aegis.acteval.metrics.BinaryRecallMetric()
        metric_object_a = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05
        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        bin_style = "perc"
        # Code to launch sampling
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        my_strata = stratification_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        [metric_object_r.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object_r,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)
        # =====================
        # # End of setup code
        # =====================

        # Do simple checks for population estimates for Accuracy
        my_strata.estimate_samples_all_systems(metric_object_a)
        sys_pop_a = my_strata.estimate_pop_all_systems(metric_object_a)
        my_strata.estimate_score_all_systems(metric_object_a)
        my_strata.estimate_score_variance_all_systems(metric_object_a)
        sys_pop_var_a = my_strata.estimate_pop_variance_all_systems(metric_object_a)
        sys_pop_frac_var_a = my_strata.estimate_pop_frac_variance_all_systems(metric_object_a)
        sys_pop_intervals_a = metric_object_a.\
            estimate_population_intervals_all_systems_strata(my_strata, alpha)

        strata_pop_list_a = [
            stratum.estimate_pop_all_systems(metric_object_a)
            for stratum in my_strata.strata
        ]
        strata_pop_frac_var_list_a = [
            stratum.estimate_pop_frac_variance_all_systems(metric_object_a)
            for stratum in my_strata.strata
        ]
        strata_pop_upper_list_a = [
            stratum.estimate_pop_upper_all_systems(metric_object_a, alpha)
            for stratum in my_strata.strata
        ]

        assert sys_pop_var_a == [0.0]
        assert sys_pop_frac_var_a == [0]
        assert sys_pop_a == [10000]
        assert sys_pop_intervals_a == [[10000.0, 10000.0, 0.0]]
        assert strata_pop_frac_var_list_a == [[0], [0], [0], [0]]
        assert strata_pop_list_a == [[2500], [2500], [2500], [2500]]
        # We cannot test direct equality because estimates may be fractions of numbers
        assert [y for x in strata_pop_list_a for y in x] == \
               [int(y) for x in strata_pop_upper_list_a for y in x]

        # Get estimates for recall metric
        my_strata.estimate_samples_all_systems(metric_object_r)
        sys_trials_scored = [stratum.get_combined_systems_score_df().shape[0]
                             for stratum in my_strata.strata]
        my_strata.estimate_pop_all_systems(metric_object_r)
        my_strata.estimate_score_all_systems(metric_object_r)
        my_strata.estimate_score_variance_all_systems(metric_object_r)
        my_strata.estimate_pop_variance_all_systems(metric_object_r)
        my_strata.estimate_pop_frac_variance_all_systems(metric_object_r)
        metric_object_r.estimate_population_intervals_all_systems_strata(my_strata, alpha)

        assert sys_trials_scored == [10, 10, 10, 10]

        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object_r)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object_r)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object_r)
            for stratum in my_strata.strata
        ]
        strata_score_var_upper_list = [
            stratum.estimate_score_variance_upper_all_systems(metric_object_r)
            for stratum in my_strata.strata
        ]
        strata_pop_upper_list = [
            stratum.estimate_pop_upper_all_systems(metric_object_r, alpha)
            for stratum in my_strata.strata
        ]

        assert strata_score_var_list == [[np.nan], [np.nan],
                                         [0.004318181818181818], [0.003615702479338843]]
        assert strata_score_var_upper_list == [[0.24909963985594238],
                                               [0.24909963985594238],
                                               [0.004318181818181818],
                                               [0.003615702479338843]]
        assert strata_pop_list == [[0], [0], [2250], [2500]]
        assert strata_pop_upper_list == [[294.635072825922],
                                         [294.635072825922],
                                         [2783.955308849338],
                                         [3022.33215691892]]

        # Take a round of adaptive sampling with recall
        trial_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            my_strata, num_success_rounds_required
        )
        assert strata_samples_list == [[0], [0], [9], [10]]
        samples = trial_sampler.draw_samples(num_step_samples, metric_object_r, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [8, 12, 13, 17]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (90, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [18, 22, 23, 27]
        # Check that all of the examples went to the required stratum objects
        my_strata.estimate_samples_all_systems(metric_object_r)
        my_strata.estimate_pop_all_systems(metric_object_r)
        my_strata.estimate_score_all_systems(metric_object_r)
        my_strata.estimate_score_variance_all_systems(metric_object_r)
        my_strata.estimate_pop_variance_all_systems(metric_object_r)
        my_strata.estimate_score_variance_upper_all_systems(metric_object_r, alpha)
        sys_conf = my_strata.get_confidence_intervals_all_systems(metric_object_r, alpha)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object_r
        )
        assert sys_conf[0][2] == 0.08872508570110604
        assert not succ_round

    def test_s1_sampler_sizes(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40
        num_success_rounds_required = 1
        num_step_samples = 50
        delta = 0.05

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "equal"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        my_strata = stratification_type(num_strata, system_list)
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [4162, 4049, 1117, 672]

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)
            num_requested_init_samples += len(initial_samples)

        # Check that each strata has the desired number of samples taken from it
        samples_taken = [stratum.get_combined_systems_score_df().shape[0]
                         for stratum in my_strata.strata]
        assert samples_taken == [10, 10, 10, 10]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [40]
        assert sys_pop[0] == 10000
        assert sys_score == [pytest.approx(0.9312097347398838)]
        assert sys_score_var == [0.0037376791377709408]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: Uniform
        trial_sampler = aegis.acteval.samplers.UniformFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        assert strata_samples_list == [[10], [10], [10], [10]]
        assert strata_pop_list == [[1117], [4049], [4162], [672]]
        assert strata_score_list == [[0.970483950182124],
                                     [0.9703204566976263],
                                     [0.8762550138492651],
                                     [0.9706337473751113]]
        assert strata_score_var_list == [[0.009561482324930438],
                                         [0.009566014254420685],
                                         [0.011640069256101082],
                                         [0.00955715713773006]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [12, 12, 13, 13]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (90, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [22, 22, 23, 23]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape == stratum_score_df.shape
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape == stratum_combined_df.shape
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert not succ_round

        # Sampler 2: Proportional Fixed
        trial_sampler = aegis.acteval.samplers.ProportionalFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [6, 20, 21, 3]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (140, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [28, 42, 44, 26]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape == stratum_score_df.shape
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape == stratum_combined_df.shape
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

        # Sampler 3: Adaptive Fixed
        trial_sampler = aegis.acteval.samplers.AdaptiveFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_list = [
            stratum.estimate_score_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_score_var_list = [
            stratum.estimate_score_variance_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        assert strata_samples_list == [[28], [42], [44], [26]]
        assert strata_pop_list == [[1117], [4049], [4162], [672]]
        assert strata_score_list == [[0.99349955040399],
                                     [0.9491143831834377],
                                     [0.9740578018288547],
                                     [0.992842065165568]]
        assert strata_score_var_list == [[0.0017296187582996115],
                                         [0.0016016463486064766],
                                         [0.0010689985508271003],
                                         [0.001994504195147037]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [6, 22, 18, 4]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (190, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [34, 64, 62, 30]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape == stratum_score_df.shape
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape == stratum_combined_df.shape
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert succ_round

    def test_three_system_known_case_two_stratify_multi(self):
        """
        Known case with three systems to test baseline stratification

        Returns: Nothing

        """
        desired_seed = 5314
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s1", "s2", "s3"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        # # Test with 2 strata per system
        num_strata = 8
        my_strata = aegis.acteval.strata.StrataMultiSystemIntersect(num_strata, system_list)
        combined_df = my_strata.get_combined_systems_df()
        assert combined_df.shape == (10000, 4)
        my_strata.stratify(bin_style="perc")
        assert my_strata.key_df.shape == (10000, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (10000, 3)
        # Check modification of combined_df
        assert combined_df.shape == (10000, 6)
        assert my_strata.num_trials == 10000
        assert my_strata.num_strata == 8
        assert my_strata.num_strata_per_system == 2
        my_strata.add_samples_to_strata(init_df)
        combined_df = my_strata.get_combined_systems_df()
        assert combined_df.shape == (10000, 6)
        # Test initial samples
        # First, check that the total number of non missing keys is the same as the number of
        # trials in the initial data frame, which is 200
        strata_key_df = combined_df.loc[
            pd.notna(combined_df["key"]), :
        ]
        assert strata_key_df.shape[0] == 200
        assert my_strata.num_trials_sampled == 200
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
        assert sys_samples == [200, 200, 200]
        assert sys_pop[0] == 10000
        assert sys_pop[1] == 10000
        assert sys_pop[2] == 10000
        assert sys_score == [0.9439438837166544, 0.7608083530744606, 0.22329881262219053]
        assert sys_score_var == [0.00028620410324043527, 0.00028620410324043516,
                                 0.0002862041032404353]

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
        assert system_conf_values == [0.027826922899548356, 0.027826922899548356,
                                      0.027826922899548356]
        my_strata.get_confidence_intervals_all_systems(my_metric, my_alpha)
        succ_conf_value = my_strata.aggregate_system_confidence_values()
        succ_round = adaptive_trial_sampler.meets_confidence_criteria(
            my_strata, my_delta, my_alpha, my_metric
        )
        assert succ_conf_value == 0.027826922899548356
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
        assert sys_samples == [300, 300, 300]

    def test_multi_strata_intersect_with_decision_s1s2(self):
        desired_seed = 1289438347
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
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
        initial_samples_requested = 160
        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        sampler_type = aegis.acteval.samplers.AdaptiveTrialSampler
        # Start with Proportional Stratification
        bin_style = "perc"

        [metric_obj.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = strata_type(num_strata, system_list)
        my_strata.dirty_strata_cache()
        assert my_strata.num_strata == 16

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
            decision_values2 = stratum_combined_df['s2_dec'].unique().tolist()
            assert len(decision_values) == 1
            assert len(decision_values2) == 1

        s1_dec_vals = combined_df['s1_dec'].unique().tolist()
        s2_dec_vals = combined_df['s2_dec'].unique().tolist()
        assert s1_dec_vals == [0, 1]
        assert s2_dec_vals == [0, 1]
        for dec1, dec2 in zip(s1_dec_vals, s2_dec_vals):
            dec_df = combined_df.loc[(combined_df['s1_dec'] == dec1) &
                                     (combined_df['s2_dec'] == dec2), :]
            stratum_index_list = dec_df['stratum_index'].value_counts().index.to_list()
            assert len(stratum_index_list) == 4

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

        assert num_total_samples == 160

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
        assert my_strata.num_strata == 16
        assert num_total_samples == 260
        assert succ_round

    def test_s1_quick_experiment(self):
        desired_seed = 18126
        np.random.seed(seed=desired_seed)
        input_dir = "data/test/sae_test_2"
        key_fpath = "data/test/sae_test_2/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10
        num_iterations = 1
        use_initial_df = False
        request_initial_samples = True
        initial_samples_requested = 40

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        np.random.seed(seed=desired_seed)
        summary_df, results_df = aegis.oracle.oracle.OracleScript. \
            get_experiment_result_data(oracle_ref,
                                       input_dir=input_dir,
                                       metric_object=metric_object,
                                       stratification_type=stratification_type,
                                       num_strata=num_strata,
                                       system_ordering=system_ordering,
                                       num_success_rounds_required=num_success_rounds_required,
                                       num_step_samples=num_step_samples, alpha=alpha,
                                       delta=delta,
                                       num_iterations=num_iterations,
                                       use_initial_df=use_initial_df,
                                       request_initial_samples=request_initial_samples,
                                       initial_samples=initial_samples_requested,
                                       parallelize=False, random_seed=desired_seed)
        # # Checks on summary_df
        summary_df_colnames = ['sampler_type', "sampler_category", 'stratification_type',
                               "bin_style", "uses_multinomial",
                               "avg_relevant_samples", "avg_samples",
                               'avg_score_diff', "perc_within_ci",
                               "perc_within_delta", 'var_score_diff',
                               "total_relevant_samples", "total_samples", "total_score_diff",
                               "num_within_ci",
                               "num_within_delta", "total_score_diff_sq",
                               "metric_name", "dataset", "system_ordering"]
        assert summary_df.shape == (16, 20)
        assert summary_df.columns.to_list() == summary_df_colnames
        # # Checks on report_df
        assert results_df.shape == (16, 38)
        min_act_system_score = min(results_df['actual_system_score'])
        max_act_system_score = max(results_df['actual_system_score'])
        assert min_act_system_score == max_act_system_score
        assert min_act_system_score == 0.9503

        # Run again without parallelization and check that we have the same scores
        np.random.seed(seed=desired_seed)
        summary_b_df, results_b_df = aegis.oracle.oracle.OracleScript. \
            get_experiment_result_data(oracle_ref,
                                       input_dir=input_dir,
                                       metric_object=metric_object,
                                       stratification_type=stratification_type,
                                       num_strata=num_strata,
                                       system_ordering=system_ordering,
                                       num_success_rounds_required=num_success_rounds_required,
                                       num_step_samples=num_step_samples, alpha=alpha,
                                       delta=delta,
                                       num_iterations=num_iterations,
                                       use_initial_df=use_initial_df,
                                       request_initial_samples=request_initial_samples,
                                       initial_samples=initial_samples_requested,
                                       parallelize=False, random_seed=desired_seed)
        assert summary_b_df.shape == (16, 20)
        assert summary_b_df.columns.to_list() == summary_df_colnames
        # # Checks on report_df
        assert results_b_df.shape == (16, 38)
        min_act_system_score = min(results_b_df['actual_system_score'])
        max_act_system_score = max(results_b_df['actual_system_score'])
        assert min_act_system_score == max_act_system_score
        assert min_act_system_score == 0.9503
        assert summary_df.equals(summary_b_df)

        summary_par_df, results_par_df = aegis.oracle.oracle.OracleScript. \
            get_experiment_result_data(oracle_ref,
                                       input_dir=input_dir,
                                       metric_object=metric_object,
                                       stratification_type=stratification_type,
                                       num_strata=num_strata,
                                       system_ordering=system_ordering,
                                       num_success_rounds_required=num_success_rounds_required,
                                       num_step_samples=num_step_samples, alpha=alpha,
                                       delta=delta,
                                       num_iterations=num_iterations,
                                       use_initial_df=use_initial_df,
                                       request_initial_samples=request_initial_samples,
                                       initial_samples=initial_samples_requested,
                                       parallelize=True, random_seed=desired_seed)
        # # Checks on summary_df
        assert summary_par_df.shape == (16, 20)
        assert summary_par_df.columns.to_list() == summary_df_colnames
        # # Checks on report_df
        assert results_par_df.shape == (16, 38)
        min_act_system_score = min(results_par_df['actual_system_score'])
        max_act_system_score = max(results_par_df['actual_system_score'])
        assert min_act_system_score == max_act_system_score
        assert min_act_system_score == 0.9503


class TestSubmissionSaeTest2WithDecisionStratification(object):

    def test_s1_decision_test1(self):
        key_fpath = "data/test/sae_test_2/key.csv"
        input_dir = "data/test/sae_test_2"
        system_ordering = ["s1"]

        # Experiment parameters
        random_seed = 5314
        np.random.seed(seed=random_seed)
        rng = np.random.RandomState(random_seed)

        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        num_strata = 4
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.01
        request_initial_samples = True
        initial_samples_requested = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)

        [metric_obj.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata = strata_type(num_strata, system_list)
        my_strata.dirty_strata_cache()

        my_strata.stratify(bin_style)
        trial_sampler = aegis.acteval.samplers.UniformFixedTrialSampler(
            my_strata, num_success_rounds_required
        )

        # Request initial samples
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_obj,
                                                      initial_samples_requested, rng)
            annotations_df = oracle_ref.get_annotations(initial_samples)
            my_strata.add_samples_to_strata(annotations_df)

        # Now sample three rounds
        # Round 1
        samples = trial_sampler.draw_samples(
            num_step_samples, metric_obj, rng
        )
        annotations_df = oracle_ref.get_annotations(samples)
        my_strata.add_samples_to_strata(annotations_df)
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_obj
        )
        assert not succ_round

        # Round 2
        trial_sampler = aegis.acteval.samplers.ProportionalTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(
            num_step_samples, metric_obj, rng
        )
        annotations_df = oracle_ref.get_annotations(samples)
        my_strata.add_samples_to_strata(annotations_df)
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_obj
        )
        assert not succ_round

        # Round 3
        trial_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(
            num_step_samples, metric_obj, rng
        )
        annotations_df = oracle_ref.get_annotations(samples)
        my_strata.add_samples_to_strata(annotations_df)
        my_strata.estimate_samples_all_systems(metric_obj)
        my_strata.estimate_pop_all_systems(metric_obj)
        my_strata.estimate_score_all_systems(metric_obj)
        my_strata.estimate_score_variance_all_systems(metric_obj)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_obj
        )
        assert not succ_round

    def test_s1_decision_experiment_completion(self):
        key_fpath = "data/test/sae_test_2/key.csv"
        input_dir = "data/test/sae_test_2"
        system_ordering = ["s1"]

        # Experiment parameters
        random_seed = 5314

        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystemDecision
        num_strata = 4
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.01
        num_iterations = 1
        use_initial_df = False
        request_initial_samples = True
        initial_samples_requested = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

        # This checks that we do not crash when trying to run random sampler with just one
        # stratum.
        (summary_out_df, results_out_df) = aegis.oracle.oracle.OracleScript. \
            get_experiment_result_data(oracle_ref,
                                       input_dir=input_dir,
                                       metric_object=metric_object,
                                       stratification_type=stratification_type,
                                       num_strata=num_strata,
                                       system_ordering=system_ordering,
                                       num_success_rounds_required=num_success_rounds_required,
                                       num_step_samples=num_step_samples, alpha=alpha,
                                       delta=delta,
                                       num_iterations=num_iterations,
                                       use_initial_df=use_initial_df,
                                       request_initial_samples=request_initial_samples,
                                       initial_samples=initial_samples_requested,
                                       parallelize=True,
                                       random_seed=random_seed)

        assert summary_out_df.shape == (16, 20)
        assert results_out_df.shape == (16, 38)
