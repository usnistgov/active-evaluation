import aegis.acteval.data_processor
import aegis.acteval.strata


class TestStratificationConstruction(object):
    """
    Pytest class to test different constructions of stratification
    """

    def test_stratify_first_s1(self):
        """
        Tests the class StrataFirstSystem for construction of strata according to the first
        system.

        Returns: Nothing.

        """
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        # threshold_df = my_data_processor.process_thresholds_data(threshold_fpaths)
        # # Test with 4 strata
        num_strata = 4
        my_strata_4 = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        assert my_strata_4.system_list[0].system_df.shape == (10000, 2)
        assert my_strata_4.system_list[1].system_df.shape == (10000, 2)
        # combined_df = my_strata_4.get_combined_systems_df()
        # assert combined_df.shape == (10000, 5)
        assert my_strata_4.system_list[0].system_df.shape == (10000, 2)
        assert my_strata_4.system_list[1].system_df.shape == (10000, 2)
        my_strata_4.stratify()
        assert my_strata_4.system_list[0].system_df.shape == (10000, 3)
        assert my_strata_4.system_list[1].system_df.shape == (10000, 3)
        combined_df = my_strata_4.get_combined_systems_df()
        # Check modification of combined_df
        assert combined_df.shape == (10000, 5)
        # Check that we have four strata with the right index numbers
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
            1,
            2,
            3,
        ]
        # Check that the strata objects are unique
        for si in range(0, len(my_strata_4.strata)):
            for sj in range(0, len(my_strata_4.strata)):
                if si != sj:
                    assert my_strata_4.strata[si] != my_strata_4.strata[sj]
        # # Now Test with 3 strata
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        num_strata = 3
        my_strata_3 = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata_3.stratify()
        combined_df = my_strata_3.get_combined_systems_df()
        assert combined_df.shape == (10000, 5)
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
            1,
            2,
        ]
        for si in range(0, len(my_strata_3.strata)):
            for sj in range(0, len(my_strata_3.strata)):
                if si != sj:
                    assert my_strata_3.strata[si] != my_strata_3.strata[sj]

    def test_stratify_first_s2(self):
        """
        Tests the class StrataFirstSystem for construction of strata according to the first
        system.

        Returns: Nothing.

        """
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        my_ordering = ["s2", "s1"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, my_ordering)
        trial_df = my_data_processor.process_trial_data(trial_data_fpath)
        assert (trial_df is None)
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        # threshold_df = my_data_processor.process_thresholds_data(threshold_fpaths)
        # # Test with 4 strata
        num_strata = 4
        my_strata_4 = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        combined_df = my_strata_4.get_combined_systems_df()
        assert combined_df.shape == (10000, 3)
        my_strata_4.stratify()
        # We should have 4 strata, but three of them empty
        assert my_strata_4.num_original_strata == 4
        assert my_strata_4.num_strata == 1
        combined_df = my_strata_4.get_combined_systems_df()
        # Check modification of combined_df
        assert combined_df.shape == (10000, 5)
        # Check that we have only one stratum index
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0,
        ]
        # Check that the strata objects are unique
        for si in range(0, len(my_strata_4.strata)):
            for sj in range(0, len(my_strata_4.strata)):
                if si != sj:
                    assert my_strata_4.strata[si] != my_strata_4.strata[sj]
        # # Now Test with 3 strata
        system_list = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                        threshold_fpaths)
        num_strata = 3
        my_strata_3 = aegis.acteval.strata.StrataFirstSystem(num_strata, system_list)
        my_strata_3.stratify()
        combined_df = my_strata_3.get_combined_systems_df()
        assert my_strata_3.num_original_strata == 3
        assert my_strata_3.num_strata == 1
        assert combined_df.shape == (10000, 5)
        assert sorted(list(combined_df["stratum_index"].value_counts().index)) == [
            0
        ]
        for si in range(0, len(my_strata_3.strata)):
            for sj in range(0, len(my_strata_3.strata)):
                if si != sj:
                    assert my_strata_3.strata[si] != my_strata_3.strata[sj]
