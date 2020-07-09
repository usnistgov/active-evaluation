import numpy as np
import pytest
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestPrecisionSubmission(object):
    """
    Class of tests specifically for data set data/test/precision_tests
    Also checks the case where bin cutoffs are dominated by one number leading to bad cutoffs
    """

    def test_precision_sampling_1(self):
        desired_seed = 42
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/precision_tests"
        key_fpath = "data/test/precision_tests/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 30

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
        assert my_strata.num_strata == 3
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert strata_size_counts == [1747, 625, 128]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]

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
        assert samples_taken == [0, 0, 10]

        # These should not crash
        sys_samples_prec = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop_prec = my_strata.estimate_pop_all_systems(metric_object)
        sys_score_prec = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var_prec = my_strata.estimate_score_variance_all_systems(metric_object)

        assert sys_samples_prec == [10]
        assert sys_pop_prec[0] == 498
        assert sys_score_prec == [pytest.approx(0.9552949798460975)]
        assert sys_score_var_prec == [0.003550226981725061]

        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()

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
        assert samples_taken == [10, 10, 10]

        # These should not crash
        sys_samples_acc = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop_acc = my_strata.estimate_pop_all_systems(metric_object)
        sys_score_acc = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var_acc = my_strata.estimate_score_variance_all_systems(metric_object)

        assert sys_samples_acc == [30]
        assert sys_pop_acc[0] == 2500
        assert sys_score_acc == [pytest.approx(0.7733315639335955)]
        assert sys_score_var_acc == [0.007617091505110385]

    def test_precision_run_initialization(self):
        desired_seed = 18176
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.01
        request_initial_samples = True
        initial_samples_requested = 400
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        key_fpath = "data/math1/math1_p5_50000t/key.csv"
        input_dir = "data/math1/math1_p5_50000t"
        system_ordering = ["s1", "s2", "s3"]
        num_strata = 8
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        sampler_type = aegis.acteval.samplers.AdaptiveFixedTrialSampler

        bin_style = "equal"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)

        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        # First, just initialize and check results
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata,
                             stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=sampler_type,
                             bin_style=bin_style,
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng,
                                      total_runs=0)
        assert not np.isnan(my_report.system_list[0].score)
        assert not np.isnan(my_report.system_list[1].score)
        assert not np.isnan(my_report.system_list[2].score)

        assert not np.isnan(my_report.system_list[0].score_variance)
        assert not np.isnan(my_report.system_list[1].score_variance)
        assert not np.isnan(my_report.system_list[2].score_variance)

    def test_precision_run_variance(self):
        desired_seed = 18176
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.01
        request_initial_samples = True
        initial_samples_requested = 400
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        key_fpath = "data/math1/math1_p5_50000t/key.csv"
        input_dir = "data/math1/math1_p5_50000t"
        system_ordering = ["s1", "s2", "s3"]
        num_strata = 8
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric()
        stratification_type = aegis.acteval.strata.StrataMultiSystemIntersect
        sampler_type = aegis.acteval.samplers.AdaptiveFixedTrialSampler

        bin_style = "equal"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)

        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        # First, just initialize and check results
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata,
                             stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=sampler_type,
                             bin_style=bin_style,
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng,
                                      total_runs=2)
        two_r_score_list = [system.score for system in my_report.system_list]
        two_r_score_var_list = [system.score_variance for system in my_report.system_list]
        assert not np.isnan(two_r_score_list[0])
        assert not np.isnan(two_r_score_list[1])
        assert not np.isnan(two_r_score_list[2])
        assert not np.isnan(two_r_score_var_list[0])
        assert not np.isnan(two_r_score_var_list[1])
        assert not np.isnan(two_r_score_var_list[2])

        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        # First, just initialize and check results
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata,
                             stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=sampler_type,
                             bin_style=bin_style,
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng,
                                      total_runs=5)
        five_r_score_list = [system.score for system in my_report.system_list]
        five_r_score_var_list = [system.score_variance for system in my_report.system_list]
        assert not np.isnan(five_r_score_list[0])
        assert not np.isnan(five_r_score_list[1])
        assert not np.isnan(five_r_score_list[2])
        assert not np.isnan(five_r_score_var_list[0])
        assert not np.isnan(five_r_score_var_list[1])
        assert not np.isnan(five_r_score_var_list[2])
        assert five_r_score_var_list[0] <= two_r_score_var_list[0]
        assert five_r_score_var_list[1] <= two_r_score_var_list[1]
        assert five_r_score_var_list[2] <= two_r_score_var_list[2]

    def test_precision_run_variance_decision(self):
        desired_seed = 18176
        num_success_rounds_required = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.01
        request_initial_samples = True
        initial_samples_requested = 400
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        key_fpath = "data/math1/math1_p5_50000t/key.csv"
        input_dir = "data/math1/math1_p5_50000t"
        system_ordering = ["s1", "s2", "s3"]
        num_strata = 8
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric()
        stratification_type = aegis.acteval.strata.StrataMultiSystemIntersectDecision
        sampler_type = aegis.acteval.samplers.AdaptiveTrialSampler

        bin_style = "equal"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)

        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        # First, just initialize and check results
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata,
                             stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=sampler_type,
                             bin_style=bin_style,
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng,
                                      total_runs=2)
        two_r_score_list = [system.score for system in my_report.system_list]
        two_r_score_var_list = [system.score_variance for system in my_report.system_list]
        assert not np.isnan(two_r_score_list[0])
        assert not np.isnan(two_r_score_list[1])
        assert not np.isnan(two_r_score_list[2])
        assert not np.isnan(two_r_score_var_list[0])
        assert not np.isnan(two_r_score_var_list[1])
        assert not np.isnan(two_r_score_var_list[2])

        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        # First, just initialize and check results
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata,
                             stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=sampler_type,
                             bin_style=bin_style,
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      my_oracle, my_experiment, rng=rng,
                                      total_runs=5)
        five_r_score_list = [system.score for system in my_report.system_list]
        five_r_score_var_list = [system.score_variance for system in my_report.system_list]
        assert not np.isnan(five_r_score_list[0])
        assert not np.isnan(five_r_score_list[1])
        assert not np.isnan(five_r_score_list[2])
        assert not np.isnan(five_r_score_var_list[0])
        assert not np.isnan(five_r_score_var_list[1])
        assert not np.isnan(five_r_score_var_list[2])
        assert five_r_score_var_list[0] <= two_r_score_var_list[0]
        assert five_r_score_var_list[1] <= two_r_score_var_list[1]
        assert five_r_score_var_list[2] <= two_r_score_var_list[2]
