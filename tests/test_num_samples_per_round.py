import numpy as np
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestNumberOfSamplesPerRound(object):
    """
    Class of tests looking at the trial samplers
    """

    def test_proportional_trial_sampler(self):
        """
        Test the ProportionalTrialSampler for scores, variance, and bins

        """
        desired_seed = 1289438347
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
        initial_samples = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment_fifty = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)

        my_controller_fifty = aegis.acteval.controller.Controller()
        my_report_fifty = my_controller_fifty.run(None, trial_data_fpath,
                                                  system_fpaths, threshold_fpaths,
                                                  oracle_ref, my_experiment_fifty, rng=rng,
                                                  total_runs=total_rounds)

        total_rounds = 1
        num_step_samples = 100

        my_experiment_hundred = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.ProportionalFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)

        my_controller_hundred = aegis.acteval.controller.Controller()
        my_report_hundred = my_controller_hundred.run(None, trial_data_fpath,
                                                      system_fpaths, threshold_fpaths,
                                                      oracle_ref, my_experiment_hundred, rng=rng,
                                                      total_runs=total_rounds)
        # assert my_report_fifty.system_list[0].score
        # == my_report_hundred.system_list[0].score
        # assert my_report_fifty.system_list[0].score_variance
        # == my_report_hundred.system_list[0].score_variance
        assert my_report_fifty.num_rounds == 2
        assert my_report_hundred.num_rounds == 1
        assert my_report_fifty.total_sampled_trials == my_report_hundred.total_sampled_trials
        assert my_report_hundred.num_samples_per_stratum == [50, 50, 50, 50]

    def test_adaptive_trial_sampler(self):
        """
        Test the ProportionalTrialSampler for scores, variance, and bins

        """
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

        num_strata = 4
        strata_type = aegis.acteval.strata.StrataFirstSystemDecision
        metric_obj = aegis.acteval.metrics.BinaryAccuracyMetric()
        total_rounds = 2
        num_step_samples = 50
        alpha = 0.05
        delta = 0.10

        # Start with Adaptive Stratification

        request_initial_samples = True
        initial_samples = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)

        my_experiment_fifty = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)

        my_controller_fifty = aegis.acteval.controller.Controller()
        my_report_fifty = my_controller_fifty.run(None, trial_data_fpath,
                                                  system_fpaths, threshold_fpaths,
                                                  oracle_ref, my_experiment_fifty, rng=rng,
                                                  total_runs=total_rounds)

        total_rounds = 1
        num_step_samples = 100

        my_experiment_hundred = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=2,
                             num_strata=num_strata, stratification_type=strata_type,
                             metric_object=metric_obj,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)

        my_controller_hundred = aegis.acteval.controller.Controller()
        my_report_hundred = my_controller_hundred.run(None, trial_data_fpath,
                                                      system_fpaths, threshold_fpaths,
                                                      oracle_ref, my_experiment_hundred, rng=rng,
                                                      total_runs=total_rounds)
        assert my_report_fifty.num_rounds == 2
        assert my_report_hundred.num_rounds == 1
        assert my_report_fifty.total_sampled_trials == my_report_hundred.total_sampled_trials
        assert my_report_hundred.num_samples_per_stratum == [48, 63, 49, 40]
