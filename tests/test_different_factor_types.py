import numpy as np
import pytest
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestFactorTypes(object):
    """
    Class of tests specifically for testing if we can give it non factor variables of 0 and 1
    """

    def test_precision_factors_string(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/factor_var_test"
        key_fpath = "data/test/factor_var_test/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric(key_values=["consonant",
                                                                                "vowel"])
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40

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
        assert strata_size_counts == [250, 250, 250, 250]

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
        assert samples_taken == [0, 0, 10, 10]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        sys_actual_score = oracle_ref.get_actual_score_all_systems(system_fpaths, threshold_fpaths,
                                                                   metric_object)

        assert sys_samples == [20]
        assert sys_pop[0] == 493
        assert sys_score == [pytest.approx(0.2263607639958344)]
        assert sys_score_var == [0.007050709955125235]
        assert sys_actual_score == [0.20081135902636918]

    def test_precision_factors_nums(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        system_ordering = ["simple"]
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric(key_values=[-1,
                                                                                1])
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40

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
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [20]

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
        assert samples_taken == [8]

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        sys_actual_score = oracle_ref.get_actual_score_all_systems(system_fpaths, threshold_fpaths,
                                                                   metric_object)

        assert sys_samples == [8]
        assert sys_pop[0] == 8
        assert sys_score == [pytest.approx(0.75)]
        assert sys_score_var == [0.0]
        assert sys_actual_score == [0.75]

    def test_accuracy_factors_string(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/factor_var_test"
        key_fpath = "data/test/factor_var_test/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric(key_values=["consonant",
                                                                               "vowel"])
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40

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
        assert strata_size_counts == [250, 250, 250, 250]

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

        # These should not crash
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        sys_actual_score = oracle_ref.get_actual_score_all_systems(system_fpaths, threshold_fpaths,
                                                                   metric_object)

        assert sys_samples == [40]
        assert sys_pop[0] == 1000
        assert sys_score == [pytest.approx(0.5235635746376782)]
        assert sys_score_var == [0.0032659033255555327]
        assert sys_actual_score == [0.49]

    def test_accuracy_factors_nums(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        system_ordering = ["simple"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric(key_values=[-1, 1])
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40

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
        assert my_strata.num_strata == 1
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [20]

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
        sys_actual_score = oracle_ref.get_actual_score_all_systems(system_fpaths, threshold_fpaths,
                                                                   metric_object)

        assert sys_samples == [20]
        assert sys_pop[0] == 20
        assert sys_score == [0.6]
        assert sys_score_var == [0.0]
        assert sys_actual_score == [0.6]
