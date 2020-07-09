import numpy as np
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestFixedSamplers(object):
    """
    Class of tests to test fixed samplers having correct amount of samples
    """

    def test_proportional_fixed_trial_sampler(self):
        """
        Test the ProportionalFixedTrialSampler to make sure its producing the
        correct amount of samples

        """
        desired_seed = 322
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 600
        assert sum(my_report.num_samples_per_stratum) == 600
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 1
        assert my_report.num_samples_per_stratum == [150, 150, 150, 150]

        desired_seed = 123489
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 500
        assert sum(my_report.num_samples_per_stratum) == 500
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 3
        assert my_report.num_samples_per_stratum == [124, 124, 126, 126]

    def test_uniform_fixed_trial_sampler(self):
        """
        Test the UniformFixedTrialSampler to make sure its producing the correct
        amount of samples

        """
        desired_seed = 322
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.UniformFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 600
        assert sum(my_report.num_samples_per_stratum) == 600
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 1
        assert my_report.num_samples_per_stratum == [150, 150, 150, 150]

        desired_seed = 123489
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.UniformFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 500
        assert sum(my_report.num_samples_per_stratum) == 500
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 3
        assert my_report.num_samples_per_stratum == [124, 124, 126, 126]

    def test_adaptive_fixed_trial_sampler(self):
        """
        Test the AdaptiveFixedTrialSampler to make sure its producing the
        correct amount of samples

        """
        desired_seed = 322
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 600
        assert sum(my_report.num_samples_per_stratum) == 600
        assert my_report.num_samples_per_stratum == [130, 179, 170, 121]

        desired_seed = 123489
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 400

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 500
        assert sum(my_report.num_samples_per_stratum) == 500
        assert my_report.num_samples_per_stratum == [114, 133, 139, 114]

    def test_proportional_fixed_trial_sampler_uneven(self):
        """
        Test the ProportionalFixedTrialSampler to make sure its producing the
        correct amount of samples

        """
        desired_seed = 322
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 402

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 602
        assert sum(my_report.num_samples_per_stratum) == 602
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 3
        assert my_report.num_samples_per_stratum == [151, 151, 150, 150]

        desired_seed = 123489
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 398

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 498
        assert sum(my_report.num_samples_per_stratum) == 498
        assert my_report.num_samples_per_stratum == [123, 123, 126, 126]

    def test_uniform_fixed_trial_sampler_uneven(self):
        """
        Test the UniformFixedTrialSampler to make sure its producing the correct
        amount of samples

        """
        desired_seed = 322
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 403

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.UniformFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 603
        assert sum(my_report.num_samples_per_stratum) == 603
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 3
        assert my_report.num_samples_per_stratum == [150, 151, 151, 151]

        desired_seed = 123489
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 397

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.UniformFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 497
        assert sum(my_report.num_samples_per_stratum) == 497
        assert (max(my_report.num_samples_per_stratum) - min(my_report.num_samples_per_stratum)) < 3
        assert my_report.num_samples_per_stratum == [124, 123, 125, 125]

    def test_adaptive_fixed_trial_sampler_uneven(self):
        """
        Test the AdaptiveFixedTrialSampler to make sure its producing the
        correct amount of samples

        """
        desired_seed = 322
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 403

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 603
        assert sum(my_report.num_samples_per_stratum) == 603
        assert my_report.num_samples_per_stratum == [130, 181, 170, 122]

        desired_seed = 123489
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        request_initial_samples = True
        initial_samples_request = 397

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples_request)

        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng,
                                      total_runs=total_rounds)

        assert my_report.total_sampled_trials == 497
        assert sum(my_report.num_samples_per_stratum) == 497
        assert my_report.num_samples_per_stratum == [114, 130, 140, 113]
