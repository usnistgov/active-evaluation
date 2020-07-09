import numpy as np
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestSmallDatasetGetExperimentResultData(object):
    """
    Pytest class that run through different steps, using a known case as a focal point.
    Each method will use one submission to test different features and evaluate.
    """

    def test_small_controller_run_one(self):
        desired_seed = 322
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["simple"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_step_samples = 50
        alpha = 0.05
        delta = 0.20
        request_initial_samples = True
        initial_samples_requested = 40

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested,
                             )

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=np.inf)
        assert my_report.num_rounds == 2
        assert my_report.total_sampled_trials == 20

    def test_small_controller_run_random_one(self):
        desired_seed = 322
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["simple"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)

        num_strata = 1
        strata_type = aegis.acteval.strata.StrataFirstSystem
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_step_samples = 50
        alpha = 0.05
        delta = 0.20
        request_initial_samples = True
        initial_samples_requested = 10

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.UniformTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested,
                             )

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=np.inf)
        assert my_report.num_rounds == 2
        assert my_report.total_sampled_trials == 20

    def test_small_controller_run_dec_random_one(self):
        desired_seed = 322
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["simple"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)

        num_strata = 1
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_step_samples = 50
        alpha = 0.05
        delta = 0.20
        request_initial_samples = True
        initial_samples_requested = 10

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.UniformTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_requested,
                             )

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(init_fpath, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=np.inf)
        assert my_report.num_rounds == 2
        assert my_report.total_sampled_trials == 20

    def test_small_experiment_one(self):
        """
        A simple case of calling the aegis.oracle.oracle.OracleScript.get_experiment_data()

        """
        desired_seed = 322
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["simple"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        (summary_out_df, results_out_df) = aegis.oracle.oracle.OracleScript. \
            get_experiment_result_data(oracle_ref=oracle_ref, input_dir=input_dir,
                                       metric_object=metric_obj,
                                       stratification_type=strata_type,
                                       num_strata=num_strata,
                                       system_ordering=my_ordering,
                                       num_success_rounds_required=1,
                                       num_step_samples=num_step_samples,
                                       alpha=alpha, delta=delta, num_iterations=2,
                                       use_initial_df=True, request_initial_samples=True,
                                       initial_samples=40, parallelize=False,
                                       random_seed=desired_seed)

        assert summary_out_df.shape == (16, 20)
        assert results_out_df.shape == (32, 38)
        # Check that we run the correct number of rounds
        assert results_out_df['num_rounds'].min() == 1
        assert results_out_df['num_rounds'].max() == 1

    def test_small_experiment_specific_data_one(self):
        """
        A simple case of calling the aegis.oracle.oracle.OracleScript.get_experiment_specific_data()

        """
        desired_seed = 322
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["simple"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        (summary_out_df, results_out_df) = aegis.oracle.oracle.OracleScript. \
            get_specific_experiment_result_data(
            oracle_ref=oracle_ref, input_dir=input_dir,
            sampler_types=[
                aegis.acteval.samplers.UniformTrialSampler,
                aegis.acteval.samplers.UniformFixedTrialSampler,
                aegis.acteval.samplers.ProportionalTrialSampler,
                aegis.acteval.samplers.ProportionalFixedTrialSampler,
                aegis.acteval.samplers.AdaptiveTrialSampler,
                aegis.acteval.samplers.AdaptiveFixedTrialSampler,
                aegis.acteval.samplers.RandomTrialSampler,
                aegis.acteval.samplers.RandomFixedTrialSampler],
            sampler_categories=["Uniform", "Uniform", "Proportional", "Proportional",
                                "Adaptive", "Adaptive", "Random", "Random"],
            bin_style_types=['equal', 'perc'],
            use_multinomials=[True, False, True, False, True, False, True, False],
            metric_object=metric_obj,
            stratification_type=strata_type,
            num_strata=num_strata,
            system_ordering=my_ordering,
            num_success_rounds_required=1,
            num_step_samples=num_step_samples,
            alpha=alpha, delta=delta, num_iterations=2,
            use_initial_df=True, request_initial_samples=True,
            initial_samples=40, parallelize=False,
            random_seed=desired_seed)

        assert summary_out_df.shape == (16, 20)
        assert results_out_df.shape == (32, 38)
        # Check that we run the correct number of rounds
        assert results_out_df['num_rounds'].min() == 1
        assert results_out_df['num_rounds'].max() == 1

    def test_small_experiment_specific_data_two(self):
        """
        A simple case of calling the aegis.oracle.oracle.OracleScript.get_experiment_specific_data()

        """
        desired_seed = 322
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/small_simple_test1"
        key_fpath = "data/test/small_simple_test1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["simple"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert not (init_df is None)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        (summary_out_df, results_out_df) = aegis.oracle.oracle.OracleScript. \
            get_specific_experiment_result_data(
            oracle_ref=oracle_ref, input_dir=input_dir,
            sampler_types=[
                aegis.acteval.samplers.ProportionalTrialSampler,
                aegis.acteval.samplers.ProportionalFixedTrialSampler],
            sampler_categories=["Proportional", "Proportional"],
            bin_style_types=['perc'],
            use_multinomials=[True, False],
            metric_object=metric_obj,
            stratification_type=strata_type,
            num_strata=num_strata,
            system_ordering=my_ordering,
            num_success_rounds_required=1,
            num_step_samples=num_step_samples,
            alpha=alpha, delta=delta, num_iterations=2,
            use_initial_df=True, request_initial_samples=True,
            initial_samples=40, parallelize=False,
            random_seed=desired_seed)

        assert summary_out_df.shape == (2, 20)
        assert results_out_df.shape == (4, 38)
        # Check that we run the correct number of rounds
        assert results_out_df['num_rounds'].min() == 1
        assert results_out_df['num_rounds'].max() == 1
