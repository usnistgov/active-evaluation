import aegis.acteval.data_processor
import aegis.acteval.system


class TestDataProcessor(object):
    """
    Pytest class to test data processor. Different test methods provide different checks on
    the different aegis.acteval.data_processor
    """

    def test_directory_file_extractor(self):
        """
        Test method to extract files from submission directory.

        """
        # importlib.reload(aegis.acteval.data_processor)
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        assert init_fpath == "data/test/sae_test_1/init.csv"
        assert trial_data_fpath is None
        assert sorted(system_fpaths) == [
            "data/test/sae_test_1/s1_outputs.csv",
            "data/test/sae_test_1/s2_outputs.csv",
        ]

        assert sorted(threshold_fpaths) == [
            "data/test/sae_test_1/s1_thresholds.csv",
            "data/test/sae_test_1/s2_thresholds.csv",
        ]

    def test_system_data_processor(self):
        """

        Test processing of system data files into System objects.

        """
        # First, get the proper files that we need
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        system_fpaths = [
            "data/test/sae_test_1/s1_outputs.csv",
            "data/test/sae_test_1/s2_outputs.csv",
        ]
        thresholds_fpaths = [
            "data/test/sae_test_1/s1_thresholds.csv",
            "data/test/sae_test_1/s2_thresholds.csv",
        ]
        systems = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                    thresholds_fpaths)

        assert len(systems) == 2
        assert systems[0].system_df.shape == (10000, 2)
        assert systems[1].system_df.shape == (10000, 2)
        assert systems[0].threshold_df.shape == (1, 3)
        assert systems[1].threshold_df.shape == (1, 3)

    def test_init_data_processor(self):
        """

        Test processing of data frame of initial samples.

        """
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        init_df = my_data_processor.process_init_data(init_fpath)
        assert init_df.shape == (72, 2)

    def test_data_processor(self):
        """

        First test for various methods of aegis.acteval.data_processor.

        """
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, ["s1", "s2"])
        assert init_fpath == "data/test/sae_test_1/init.csv"
        assert trial_data_fpath is None
        systems = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                    threshold_fpaths)
        assert len(systems) == 2
        assert systems[0].system_df.shape == (10000, 2)
        assert systems[1].system_df.shape == (10000, 2)
        assert systems[0].threshold_df.shape == (1, 3)
        assert systems[1].threshold_df.shape == (1, 3)
        assert systems[0].system_id == "s1"
        assert systems[1].system_id == "s2"

    def test_data_processor_orderings(self):
        """

        Tests to check that the data processor can properly take in custom orderings of systems.

        Returns: Nothing.

        """
        input_dir = "data/test/sae_test_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # no ordering
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        assert sorted(system_fpaths) == [
            "data/test/sae_test_1/s1_outputs.csv",
            "data/test/sae_test_1/s2_outputs.csv",
        ]

        assert sorted(threshold_fpaths) == [
            "data/test/sae_test_1/s1_thresholds.csv",
            "data/test/sae_test_1/s2_thresholds.csv",
        ]
        # ordering ["s1", "s2"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, ["s1", "s2"])
        assert system_fpaths == [
            "data/test/sae_test_1/s1_outputs.csv",
            "data/test/sae_test_1/s2_outputs.csv",
        ]

        assert threshold_fpaths == [
            "data/test/sae_test_1/s1_thresholds.csv",
            "data/test/sae_test_1/s2_thresholds.csv",
        ]
        # ordering ["s2", "s1"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, ["s2", "s1"])
        assert system_fpaths == [
            "data/test/sae_test_1/s2_outputs.csv",
            "data/test/sae_test_1/s1_outputs.csv",
        ]

        assert threshold_fpaths == [
            "data/test/sae_test_1/s2_thresholds.csv",
            "data/test/sae_test_1/s1_thresholds.csv",
        ]

    def test_data_processor_uniform_case(self):
        """

        Tests data processor with an underscore in the system id, s_uniform.

        """
        input_dir = "data/test/simple_uniform_1"
        my_data_processor = aegis.acteval.data_processor.DataProcessor()
        # no ordering
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir)
        assert system_fpaths == [
            "data/test/simple_uniform_1/s_uniform_outputs.csv",
        ]
        assert threshold_fpaths == [
            "data/test/simple_uniform_1/s_uniform_thresholds.csv",
        ]
        systems = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                    threshold_fpaths)
        assert systems[0].system_id == "s_uniform"
        # ordering ["s_uniform"]
        init_fpath, trial_data_fpath, system_fpaths, threshold_fpaths = \
            my_data_processor.extract_files_from_directory(input_dir, ["s_uniform"])
        assert system_fpaths == [
            "data/test/simple_uniform_1/s_uniform_outputs.csv",
        ]
        assert threshold_fpaths == [
            "data/test/simple_uniform_1/s_uniform_thresholds.csv",
        ]
        systems = my_data_processor.process_systems_with_thresholds(system_fpaths,
                                                                    threshold_fpaths)
        assert systems[0].system_id == "s_uniform"
