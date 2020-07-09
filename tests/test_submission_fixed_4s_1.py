import math
import numpy as np
import pytest
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestSubmissionFixed4s1Components(object):
    """
    Class of tests specifically for data set data/test/fixed_4s_1
    """

    def test_initialization_s1(self):
        desired_seed = 3921
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples = 40
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
        strata_alpha = my_strata.get_strata_alpha(alpha)
        assert strata_alpha == 0.05
        my_strata.stratify(bin_style=bin_style)
        # Check that we have the same number of trials within each bin
        assert my_strata.num_strata == num_strata
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert len(set(strata_size_counts)) == 1
        stratum_s1_scores = [stratum.get_combined_systems_df()['s1'].value_counts().index.to_list()
                             for stratum in my_strata.strata]
        # Check that we have the stratum by the desired scores
        assert stratum_s1_scores == [[0.1], [0.35], [0.6], [0.85]]

        # Get initial samples
        num_requested_init_samples = 0
        if request_initial_samples:
            initial_samples = \
                my_strata.find_needed_initial_samples(metric_object,
                                                      initial_samples, rng)
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
        assert sys_pop[0] == 20000
        # We have a correction for the bias in our score estimate, so it should be
        # slightly corrected. Likewise for the score variance
        assert sys_score == [pytest.approx(0.5470308622571677)]
        assert sys_score_var == [pytest.approx(0.0030483370295337478)]
        my_strata.get_confidence_intervals_all_systems(metric_object, alpha)
        sys_conf_vals = [sys_conf[2] for sys_conf in my_strata.system_confidence_list]
        assert sys_conf_vals == [0.10821303604719235]
        # Check the score and score_var estimates of each stratum
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
        assert strata_pop_list == [[5000], [5000], [5000], [5000]]
        # With seed, values checked by hand
        assert strata_score_list == [[0.9703086225716765], [0.5],
                                     [0.02969137742832344], [0.6881234490286706]]
        # These values are
        assert strata_score_var_list == [[0.009566334709863891],
                                         [0.015325042142266095],
                                         [0.009566334709863886],
                                         [0.01440364895308174]]

        # # Try something a bit fun: Sample one round with each sampler given the current
        # # strata and passing it through

        # Sampler 1: UniformFixed
        trial_sampler = aegis.acteval.samplers.UniformFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng=rng)
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
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng=rng)
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
        assert score_df.shape == (140, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [34, 34, 36, 36]
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
        assert not succ_round

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
        assert strata_samples_list == [[34], [34], [36], [36]]
        assert strata_pop_list == [[5000], [5000], [5000], [5000]]
        assert strata_score_list == [[0.9950389390822239],
                                     [0.5582398751861439],
                                     [0.004555332051037338],
                                     [0.720197630199539]]
        assert strata_score_var_list == [[0.0011620553232032148],
                                         [0.006606765278693372],
                                         [0.0010259141832022472],
                                         [0.005306117224016602]]

        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng=rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [8, 18, 7, 17]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (190, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [42, 52, 43, 53]
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
        # Now check that the score and score variance proportions are weighted properly
        # by population and not by samples
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

        assert strata_samples_list == [[42], [52], [43], [53]]
        assert strata_pop_list == [[5000], [5000], [5000], [5000]]
        assert strata_score_list == [[0.9963826044570205], [0.5956488925449992],
                                     [0.003492120646508021], [0.6971021721165758]]
        assert strata_score_var_list == [[0.000728107916653382], [0.004414239142374931],
                                         [0.0006904265132997928], [0.0038494804172937223]]

    def test_s1_comp_initial_samples(self):
        desired_seed = 18653257
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 40

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        bin_style = "perc"

        my_data_processor = aegis.acteval.data_processor.DataProcessor()

        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)

        num_initial_iterations = 20
        score_list = []
        score_var_list = []
        conf_val_list = []
        for iter_num in range(0, num_initial_iterations + 1):
            system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                            threshold_fpaths)
            my_strata = stratification_type(num_strata, system_list)
            my_strata.stratify(bin_style=bin_style)
            strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
            assert len(set(strata_size_counts)) == 1
            stratum_s1_scores = [stratum.get_combined_systems_df()['s1'].
                                 value_counts().index.to_list()
                                 for stratum in my_strata.strata]
            # Check that we have the stratum by the desired scores
            assert stratum_s1_scores == [[0.1], [0.35], [0.6], [0.85]]
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
            assert sys_pop[0] == 20000
            score_list.append(sys_score[0])
            score_var_list.append(sys_score_var[0])
            my_strata.get_confidence_intervals_all_systems(metric_object, alpha)
            sys_conf_vals = [sys_conf[2] for sys_conf in my_strata.system_confidence_list]
            conf_val_list.append(sys_conf_vals[0])
        # We apply a bias correction in our score, so it is not 0.6
        assert np.median(score_list) == pytest.approx(0.5940617245143353)
        assert np.median(score_var_list) == 0.002918999386610572
        assert np.median(conf_val_list) == 0.10589247348747655

    def test_strata_first_with_decision(self):
        desired_seed = 1289438347
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # single system
        my_ordering = ["s1"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (init_df is None)
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
            # We have exact sizes for these stratum
            assert math.fabs(strata_size_counts[0] - strata_size_counts[1]) <= 0

        #  Check that scores are in line with the decisions
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
            num_step_samples, metric_obj, rng=rng
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

    def test_recall_round_estimates(self):

        # ========================
        # # # Code to setup test
        # ========================
        desired_seed = 4201
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryRecallMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        alpha = 0.05
        num_strata = 4
        request_initial_samples = True
        initial_samples_requested = 400
        num_success_rounds_required = 1
        num_step_samples = 100
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
        # =====================
        # # End of setup code
        # =====================

        # Get estimates for recall metric
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        sys_scores = my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        my_strata.estimate_pop_variance_all_systems(metric_object)
        my_strata.estimate_pop_frac_variance_all_systems(metric_object)
        my_strata.estimate_score_variance_upper_all_systems(metric_object, alpha)
        metric_object.estimate_population_intervals_all_systems_strata(my_strata, alpha)
        sys_scores_lower = my_strata.estimate_score_lower_all_systems(metric_object, alpha)
        sys_scores_upper = my_strata.estimate_score_upper_all_systems(metric_object, alpha)
        sys_conf = my_strata.get_confidence_intervals_all_systems(metric_object, alpha)
        sys_trials_scored = [stratum.get_combined_systems_score_df().shape[0]
                             for stratum in my_strata.strata]
        sys_conf_true_pop = my_strata.get_confidence_intervals_true_pop_all_systems(metric_object,
                                                                                    alpha)
        assert sys_trials_scored == [100, 100, 100, 100]
        assert sys_scores_lower <= sys_scores
        assert sys_scores <= sys_scores_upper
        assert sys_conf_true_pop[0][2] <= sys_conf[0][2]

        strata_samples_list = [
            stratum.estimate_samples_all_systems(metric_object)
            for stratum in my_strata.strata
        ]
        strata_pop_list = [
            stratum.estimate_pop_all_systems(metric_object)
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
        strata_pop_upper_list = [
            stratum.estimate_pop_upper_all_systems(metric_object, alpha)
            for stratum in my_strata.strata
        ]
        strata_pop_lower_list = [
            stratum.estimate_pop_lower_all_systems(metric_object, alpha)
            for stratum in my_strata.strata
        ]

        assert strata_score_var_list == [[np.nan], [0.0003010670731707317],
                                         [np.nan], [8.062234206343262e-05]]
        assert strata_score_var_upper_list == [[0.2450490098019604],
                                               [0.0003010670731707317],
                                               [0.2450490098019604],
                                               [8.062234206343262e-05]]
        assert strata_pop_list == [[0], [1950], [0], [3850]]
        assert strata_pop_upper_list == [[68.10275307209747],
                                         [2321.417260191468],
                                         [68.10275307209747],
                                         [4213.166231685998]]
        assert strata_pop_lower_list <= strata_pop_list
        assert strata_pop_list <= strata_pop_upper_list

        # Take a round of adaptive sampling with recall
        trial_sampler = aegis.acteval.samplers.AdaptiveTrialSampler(
            my_strata, num_success_rounds_required
        )
        assert strata_samples_list == [[0], [39], [0], [77]]
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        assert samples_df.shape[0] == 100
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [27, 34, 25, 14]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (100, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (500, 5)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [127, 134, 125, 114]
        # Check that all of the examples went to the required stratum objects
        my_strata.estimate_samples_all_systems(metric_object)
        my_strata.estimate_pop_all_systems(metric_object)
        my_strata.estimate_score_all_systems(metric_object)
        my_strata.estimate_score_variance_all_systems(metric_object)
        my_strata.estimate_pop_variance_all_systems(metric_object)
        my_strata.estimate_score_variance_upper_all_systems(metric_object, alpha)
        sys_conf = my_strata.get_confidence_intervals_all_systems(metric_object, alpha)
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        assert sys_conf[0][2] == 0.03022196334611993
        assert succ_round


class TestSubmissionFixed4s1ExperimentRuns(object):
    """
    Class of tests specifically for data set data/test/fixed_4s_1
    """

    def test_s1_quick_controller_run(self):
        desired_seed = 18126
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        num_success_rounds_required = 3
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10
        request_initial_samples = True
        initial_samples = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=aegis.acteval.samplers.AdaptiveTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)
        my_controller = aegis.acteval.controller.Controller()
        my_report = my_controller.run(None, trial_data_fpath,
                                      system_fpaths, threshold_fpaths,
                                      oracle_ref, my_experiment, rng=rng)
        assert my_report.total_sampled_trials == 400
        assert my_report.system_list[0].score == 0.6096609187539422
        assert my_report.system_list[0].score_variance == 0.00020289728239852086
        assert my_report.system_list[0].confidence_value == 0.028554702515284247
        assert my_report.num_rounds == 3

    def test_s1_quick_controller_run_true_random_precision_test(self):
        desired_seed = 18126
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        system_ordering = ["s1"]
        metric_object = aegis.acteval.metrics.BinaryPrecisionMetric()
        stratification_type = aegis.acteval.strata.StrataFirstSystem
        num_strata = 4
        num_success_rounds_required = 3
        num_step_samples = 100
        alpha = 0.05
        delta = 0.10
        request_initial_samples = True
        initial_samples = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=aegis.acteval.samplers.TrueRandomTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)
        my_controller = aegis.acteval.controller.Controller()
        my_report_tr = my_controller.run(None, trial_data_fpath,
                                         system_fpaths, threshold_fpaths,
                                         oracle_ref, my_experiment, rng=rng)
        assert my_report_tr.total_sampled_trials == 350
        assert my_report_tr.system_list[0].score == 0.4070480895117912
        assert my_report_tr.system_list[0].score_variance == 0.0003976373904115442
        assert my_report_tr.system_list[0].confidence_value == 0.03997450847963574
        assert my_report_tr.num_rounds == 3

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=aegis.acteval.samplers.TrueRandomFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)
        my_controller = aegis.acteval.controller.Controller()
        my_report_trf = my_controller.run(None, trial_data_fpath,
                                          system_fpaths, threshold_fpaths,
                                          oracle_ref, my_experiment, rng=rng)
        assert my_report_trf.total_sampled_trials == 350
        assert my_report_trf.system_list[0].score == 0.38787297807142784
        assert my_report_trf.system_list[0].score_variance == 0.000411029629265162
        assert my_report_trf.system_list[0].confidence_value == 0.040642095275555956
        assert my_report_trf.num_rounds == 3

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=aegis.acteval.samplers.RandomTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)
        my_controller = aegis.acteval.controller.Controller()
        my_report_r = my_controller.run(None, trial_data_fpath,
                                        system_fpaths, threshold_fpaths,
                                        oracle_ref, my_experiment, rng=rng)
        assert my_report_r.total_sampled_trials == 350
        assert my_report_r.system_list[0].score == 0.3805965606676374
        assert my_report_r.system_list[0].score_variance == 0.0002481599775306146
        assert my_report_r.system_list[0].confidence_value == 0.03157951061640979
        assert my_report_r.num_rounds == 3

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, system_ordering)
        my_experiment = aegis.acteval.experiment. \
            ExperimentParams(num_step_samples=num_step_samples, alpha=alpha, delta=delta,
                             num_success_rounds_required=num_success_rounds_required,
                             num_strata=num_strata, stratification_type=stratification_type,
                             metric_object=metric_object,
                             sampler_type=aegis.acteval.samplers.RandomFixedTrialSampler,
                             bin_style="perc",
                             request_initial_samples=request_initial_samples,
                             initial_samples=initial_samples)
        my_controller = aegis.acteval.controller.Controller()
        my_report_rf = my_controller.run(None, trial_data_fpath,
                                         system_fpaths, threshold_fpaths,
                                         oracle_ref, my_experiment, rng=rng)
        assert my_report_rf.total_sampled_trials == 350
        assert my_report_rf.system_list[0].score == 0.3874384994039191
        assert my_report_rf.system_list[0].score_variance == 0.0002536214638440199
        assert my_report_rf.system_list[0].confidence_value == 0.03192511919257601
        assert my_report_rf.num_rounds == 3

        assert my_report_rf.total_sampled_trials <= \
            my_report_trf.total_sampled_trials
        assert my_report_rf.system_list[0].confidence_value <= \
            my_report_trf.system_list[0].confidence_value
        assert my_report_r.total_sampled_trials <= \
            my_report_tr.total_sampled_trials
        assert my_report_r.system_list[0].confidence_value <= \
            my_report_tr.system_list[0].confidence_value

    def test_s1_quick_experiment(self):
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
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
        initial_samples = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

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
                                       initial_samples=initial_samples,
                                       parallelize=False, random_seed=None)
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

        # Repeat tests but enable parallelization to check that parallelization
        # does not change results
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
                                       initial_samples=initial_samples,
                                       parallelize=True, random_seed=None)
        assert summary_par_df.shape == (16, 20)
        assert summary_par_df.columns.to_list() == summary_df_colnames
        # # Checks on report_df
        assert results_par_df.shape == (16, 38)

    def test_s1_quick_experiment_rng(self):
        random_seed = 18126
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
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
        initial_samples = 100

        oracle_ref = aegis.oracle.oracle.OracleScript(key_fpath)

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
                                       initial_samples=initial_samples,
                                       parallelize=False, random_seed=random_seed)
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

        # Repeat tests but enable parallelization to check that parallelization
        # does not change results
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
                                       initial_samples=initial_samples,
                                       parallelize=True,
                                       random_seed=random_seed)
        assert summary_par_df.shape == (16, 20)
        assert summary_par_df.columns.to_list() == summary_df_colnames
        # # Checks on report_df
        assert results_par_df.shape == (16, 38)

        # Repeat parallelization to check that we have the same results with the seed
        summary_par_b_df, results_par_b_df = aegis.oracle.oracle.OracleScript. \
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
                                       initial_samples=initial_samples,
                                       parallelize=True,
                                       random_seed=random_seed)
        assert summary_par_b_df.shape == (16, 20)
        assert summary_par_b_df.columns.to_list() == summary_df_colnames
        # # Checks on report_df
        assert results_par_b_df.shape == (16, 38)
        # Check that seed reproduces results
        assert summary_par_df.equals(summary_par_b_df)


class TestSubmissionFixed4s1MultiComponents(object):

    def test_initialization_s1_skey(self):
        desired_seed = 3921
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
        system_ordering = ["s1", "skey"]
        metric_object = aegis.acteval.metrics.BinaryAccuracyMetric()
        stratification_type = aegis.acteval.strata.StrataMultiSystemIntersect
        alpha = 0.05
        num_strata = 16
        request_initial_samples = True
        initial_samples_requested = 60
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
        strata_alpha = my_strata.get_strata_alpha(alpha)
        assert strata_alpha == 0.025320565519103666
        my_strata.stratify(bin_style=bin_style)
        # We have 10 empty stratum that should be removed
        assert my_strata.num_strata == 6
        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().sort_index().to_list()
        assert strata_size_counts == [5000, 5000, 4000, 1000, 1000, 4000]
        stratum_s1_scores = [stratum.get_combined_systems_df()['s1'].value_counts().index.to_list()
                             for stratum in my_strata.strata]
        # Check that we have the stratum by the desired scores
        assert stratum_s1_scores == [[0.1], [0.6], [0.35], [0.85], [0.35], [0.85]]
        stratum_skey_scores = [stratum.get_combined_systems_df()['skey'].value_counts().
                               index.to_list()
                               for stratum in my_strata.strata]
        assert stratum_skey_scores == [[0.1], [0.35], [0.45, 0.55], [0.45], [0.85], [0.85]]

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
        assert samples_taken == [10, 10, 10, 10, 10, 10]

        [metric_object.convert_thresholds_to_decisions(system) for system in system_list]
        my_strata.dirty_strata_cache()
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
        assert sys_samples == [60, 60]
        assert sys_pop[0] == 20000
        assert sys_score == [pytest.approx(0.565826052850762), 0.9703338385589526]
        assert sys_score_var == [0.0022253214101603056, 0.0020045691501327395]

        my_strata.get_confidence_intervals_all_systems(metric_object, alpha)
        sys_conf_vals = [sys_conf[2] for sys_conf in my_strata.system_confidence_list]
        assert sys_conf_vals == [0.09245799348860972, 0.0877523210147102]
        my_strata.get_confidence_intervals_all_systems(metric_object, alpha)
        aggregated_conf = my_strata.aggregate_system_confidence_values()
        aggregated_samples = my_strata.aggregate_system_stats(sys_samples)
        aggregated_pop = my_strata.aggregate_system_stats(sys_pop)
        aggregated_score = my_strata.aggregate_system_stats(sys_score)
        aggregated_score_var = my_strata.aggregate_system_stats(sys_score_var)
        assert aggregated_conf == 0.09245799348860972
        assert aggregated_samples == pytest.approx(60)
        assert aggregated_pop == pytest.approx(20000)
        assert aggregated_score == 0.9703338385589526
        assert aggregated_score_var == 0.0022253214101603056
        # Check the score and score_var estimates of each stratum
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
        assert strata_samples_list == [[10, 10], [10, 10], [10, 10],
                                       [10, 10], [10, 10], [10, 10]]
        assert strata_pop_list == [[5000, 5000],
                                   [5000, 5000],
                                   [4000, 4000],
                                   [1000, 1000],
                                   [1000, 1000],
                                   [4000, 4000]]
        # With seed, values checked by hand
        # bias correction for score values gives values that are slightly lower
        assert strata_score_list == [[0.9703086225716765, 0.9703086225716765],
                                     [0.02969137742832344, 0.9703086225716765],
                                     [0.5940642437887114, 0.9703212189435568],
                                     [0.02948960304308452, 0.9705103969569153],
                                     [0.02948960304308452, 0.9705103969569153],
                                     [0.9703212189435568, 0.9703212189435568]]
        # These values are
        assert strata_score_var_list == [[0.009566334709863891, 0.009566334709863891],
                                         [0.009566334709863886, 0.009566334709863891],
                                         [0.015097248515104597, 0.009565993578699599],
                                         [0.00956073079106985, 0.009560730791069848],
                                         [0.00956073079106985, 0.009560730791069848],
                                         [0.009565993578699599, 0.009565993578699599]]

        # Take adaptive samples for one round
        trial_sampler = aegis.acteval.samplers.AdaptiveFixedTrialSampler(
            my_strata, num_success_rounds_required
        )
        samples = trial_sampler.draw_samples(num_step_samples, metric_object, rng=rng)
        # Check stratum of samples
        combined_df = my_strata.get_combined_systems_df()
        samples_df = combined_df.loc[combined_df['trial_id'].isin(samples), :]
        # We should have plenty of samples in each stratum, but fixed takes a floor
        assert samples_df.shape[0] == 50
        samples_count = samples_df['stratum_index'].value_counts()
        samples_count.sort_index(inplace=True)
        assert samples_count.to_list() == [12, 12, 12, 2, 2, 10]
        my_oracle = aegis.oracle.oracle.OracleScript(key_fpath)
        annotations_df = my_oracle.get_annotations(samples)
        assert annotations_df.shape == (50, 2)
        my_strata.add_samples_to_strata(annotations_df)
        score_df = my_strata.get_combined_systems_score_df()
        assert score_df.shape == (110, 7)
        score_ind_count = score_df['stratum_index'].value_counts()
        score_ind_count.sort_index(inplace=True)
        assert score_ind_count.to_list() == [22, 22, 22, 12, 12, 20]
        # Check that all of the examples went to the required stratum objects
        for stratum in my_strata.strata:
            stratum_combined_df = stratum.get_combined_systems_df()
            stratum_score_df = stratum.get_combined_systems_score_df()
            assert score_df.loc[score_df['stratum_index'] ==
                                stratum.stratum_index, :].shape == stratum_score_df.shape
            assert combined_df.loc[combined_df['stratum_index'] ==
                                   stratum.stratum_index, :].shape == stratum_combined_df.shape
        # Now check that the score is the proper weighting of the stratum score
        sys_samples = my_strata.estimate_samples_all_systems(metric_object)
        sys_pop = my_strata.estimate_pop_all_systems(metric_object)
        sys_score = my_strata.estimate_score_all_systems(metric_object)
        sys_score_var = my_strata.estimate_score_variance_all_systems(metric_object)
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
        assert strata_pop_list == [[5000, 5000],
                                   [5000, 5000],
                                   [4000, 4000],
                                   [1000, 1000],
                                   [1000, 1000],
                                   [4000, 4000]]
        assert strata_score_list == [[0.9905324956626498, 0.9905324956626498],
                                     [0.009467504337350196, 0.9905324956626498],
                                     [0.7675685240317093, 0.9905422940581337],
                                     [0.022710737979794726, 0.9772892620202054],
                                     [0.022710737979794726, 0.9772892620202054],
                                     [0.9891150191597987, 0.9891150191597987]]
        assert sys_score == [0.603607782436281, 0.9889266366769318]
        assert strata_score_var_list == [[0.0028326845767593007, 0.0028326845767593007],
                                         [0.002832684576759303, 0.0028326845767593007],
                                         [0.007534661332453077, 0.0028308827843044307],
                                         [0.007635226018815964, 0.0076352260188159654],
                                         [0.007635226018815964, 0.0076352260188159654],
                                         [0.003383632649614354, 0.003383632649614354]]
        assert sys_score_var == [0.0008248599212740684, 0.0006376968198533597]
        succ_round = trial_sampler.meets_confidence_criteria(
            my_strata, delta, alpha, metric_object
        )
        conf_value = my_strata.aggregate_system_confidence_values()
        assert conf_value == 0.05629089997872949
        assert not succ_round

    def test_multi_strata_intersect_with_decision(self):
        desired_seed = 1289438347
        np.random.seed(seed=desired_seed)
        rng = np.random.RandomState(desired_seed)
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/fixed_4s_1"
        key_fpath = "data/test/fixed_4s_1/key.csv"
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
        # trials and decisions align completely for s1 and s2, so should be 4 strata
        assert my_strata.num_strata == 4

        combined_df = my_strata.get_combined_systems_df()

        # Since bin style is perc, check that each bin within the same decision has the
        # same size
        for ind in range(0, len(my_strata.strata)):
            curr_stratum = my_strata.strata[ind]
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

        strata_size_counts = my_strata.key_df['stratum_index'].value_counts().to_list()
        assert len(set(strata_size_counts)) == 1

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
            num_step_samples, metric_obj, rng=rng
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
