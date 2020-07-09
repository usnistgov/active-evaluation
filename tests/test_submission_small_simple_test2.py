import pandas as pd
import pytest
import aegis.acteval.data_processor
import aegis.acteval.strata
import aegis.acteval.metrics
import aegis.acteval.samplers
import aegis.oracle.oracle
import aegis.acteval.controller
import aegis.acteval.experiment


class TestSubmissionSmallSimpleTest2(object):
    """
    Pytest class that run through different steps, using a known case as a focal point.
    Each method will use one submission to test different features and evaluate.
    """

    def test_small_simple_test2_case_two_strata(self):
        """
        A super-simple case of 2 trials with one system, thresholded at 0.
        First is scored as -0.1 scoring it correctly,
        and the last trial is correctly scored at 0.1.
        The entire key is provided initially, key is -1 or 1. Stratified in two strata,
        equal-width bin so the stratification places all scores of -0.1 in one strata and
        scores of 0.1 in the other strata.

        Returns: Nothing.

        """
        input_dir = "data/test/small_simple_test2"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        init_df = my_data_processor.process_init_data(init_fpath)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        assert system_fpaths == ["data/test/small_simple_test2/simple_outputs.csv"]
        assert threshold_fpaths == ["data/test/small_simple_test2/simple_thresholds.csv"]
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        assert len(system_list) == 1
        # This case uses only one strata
        num_strata = 2
        my_strata = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata.stratify(bin_style='equal')
        assert my_strata.key_df.shape == (2, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (2, 3)
        # Check that the widths of the bins are equal by checking that we have the proper
        # stratum sizes

        # Check modification of combined_df
        assert combined_df.shape == (2, 4)
        assert my_strata.num_trials == 2
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
        assert my_strata.key_df.shape == (2, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert combined_df.shape == (2, 5)
        my_strata.add_samples_to_strata(init_df)
        assert my_strata.key_df.shape == (2, 3)
        combined_df = my_strata.get_combined_systems_df()
        assert my_strata.key_df.shape == (2, 3)
        assert combined_df.shape == (2, 5)
        strata_key_df = combined_df.loc[pd.notna(combined_df["key"]), :]
        assert strata_key_df.shape[0] == 2
        assert my_strata.num_trials_sampled == 2
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
        assert sys_pop[0] == 2
        # This score and variance were calculated by hand
        # score computed to be 0.6
        assert sys_score == [pytest.approx(1.0)]
        assert sys_score_var == [0]
        # Next, sample and check that we cannot get any more samples
        pass
