"""Converts between data in various formats and the desired internal formats."""
import pandas as pd
import os
import re
import aegis.acteval.system


class DataProcessor:
    """
    The data processing class of the Active Evaluator.

    To ease logistics, all of the data processing functions required of the Active Evaluator are
    placed in this one class.
    """

    def __init__(self):
        pass

    def get_system_id_from_filename(self, fpath):
        """
        Shortcut function for getting system ids from files. Useful for list comprehensions and
        is used in other methods of the data processor.
        Args:
            fpath: the filepath, which could be a filename or a directory

        Returns: the string with the system_id from the filename.

        """
        fname = os.path.basename(fpath)
        fname_tokens = re.split("_", fname)
        if len(fname_tokens) == 2:
            sys_id = fname_tokens[0]
        else:
            sys_id = '_'.join(fname_tokens[:-1])
        return sys_id

    def extract_files_from_directory(self, input_dir, system_ordering=None):
        """
        Extracts files from submission directory.

        This takes the directory, and finds the files it needs, ignoring other files in that
        directory, using filenames as the identifiers.

        Ignores additional files.

        Args:
            input_dir (str):
                the full path to the input directory
            system_ordering (list of str, optional):
                An ordering of systems by system_id. If this list is empty, the
                code reads all systems in any order. Else, it takes only the systems with the
                specified system ids and reads them into a list. If a system id is provided that
                is not in the file, it will make the file here anyway and the next method may
                return an error. If no ordering is specified, the system_fpaths give the
                order that the files are read. Defaults to None, which resolves to [].

        Returns:
            (init_fpath, metadata_fpath, system_fpaths, thresholds_fpaths):

            The tuple is,
            (init_fpath, metadata_fpath, system_fpaths, thresholds_fpaths) with:

            init_fpath
                the path to the init.csv file with ground truths for initial samples
            metadata_fpath
                the path to the trial_metadata.csv, which is the data frame with the
                trial features.
            system_fpaths
                the list of system output files, notated as <system_id>_outputs.csv.
                Files can appear in any order and need not match the ordering of the
                threshold fpaths.
            thresholds_fpaths
                the list of system threshold files, notated as
                <system_id>_thresholds.csv. The files can appear in any order.


        """
        init_fpath = None
        metadata_fpath = None
        system_fpaths = []
        thresholds_fpaths = []

        if system_ordering is None:
            system_ordering = []

        if not system_ordering:
            file_names = os.listdir(input_dir)
            for fname in file_names:
                if fname == "init.csv":
                    init_fpath = os.path.join(input_dir, fname)
                    continue
                elif fname == "trial_metadata.csv":
                    metadata_fpath = os.path.join(input_dir, fname)
                    continue
                fname_tokens = re.split("_", fname)
                # need system id and outputs, can ignore middle token
                if len(fname_tokens) < 2:
                    continue
                if fname_tokens[-1] == "outputs.csv":
                    system_fpaths.append(os.path.join(input_dir, fname))
                    sys_id = self.get_system_id_from_filename(fname)
                    # Check for corresponding threshold file
                    if os.path.isfile(os.path.join(input_dir, sys_id + "_thresholds.csv")):
                        thresholds_fpaths.append(
                            os.path.join(input_dir, sys_id + "_thresholds.csv")
                        )
        else:
            # list is not empty and is assumed to have all ids. Ignores systems not
            file_names = os.listdir(input_dir)
            for fname in file_names:
                if fname == "init.csv":
                    init_fpath = os.path.join(input_dir, fname)
                    continue
                elif fname == "trial_metadata.csv":
                    metadata_fpath = os.path.join(input_dir, fname)
                    continue
            # Now add paths by id ordering
            system_fpaths = [os.path.join(input_dir, str(sys_id) + "_outputs.csv")
                             for sys_id in system_ordering]
            thresholds_fpaths = [os.path.join(input_dir, str(sys_id) + "_thresholds.csv")
                                 for sys_id in system_ordering]

        return init_fpath, metadata_fpath, system_fpaths, thresholds_fpaths

    def process_init_data(self, init_fpath):
        """
        Processes in the data frame of initial samples.

        Args:
            init_fpath (str): The path to the initial trials file of initially-scored samples.

        Returns:
            pandas.core.frame.DataFrame: init_df, the processed inital trials as a data frame.

        """
        if init_fpath is None:
            return None
        init_df = pd.read_csv(init_fpath)
        return init_df

    def process_trial_data(self, trial_data_fpath):
        """
        Reads in and processes the trial data (features) into a data frame.

        Args:
            trial_data_fpath (str): the path to the trials metadata file.

        Returns:
            pandas.core.frame.DataFrame: trial_df, the data frame of the trials features.
            If trial_data_fpath is None, returns None

        """
        if trial_data_fpath is None:
            return None

        trial_df = pd.read_csv(trial_data_fpath)
        return trial_df

    def process_systems_with_thresholds(self, system_filepaths, threshold_filepaths):
        """
        Takes the system filepaths and the threshold filepahs, and produces system objects
        with the necessary threshold information.

        This method assumes that the system_filepaths and threshold_filepaths are in the same
        order with matching system ids. The order can be customized with the
        extract_files_from_directory method.

        Args:
            system_filepaths (list of object):
                A list of file paths, with each entry a file path to a system's
                output data.
            threshold_filepaths (list of object):
                the list of paths to the system threshold files.


        Returns:
            list of aegis.acteval.system.System: sys_list,
            a list of systems with their information as System objects.

        """

        system_data_frames = [
            pd.read_csv(system_path) for system_path in system_filepaths
        ]
        system_ids = [
            self.get_system_id_from_filename(fname) for fname in system_filepaths
        ]
        threshold_data_frames = [
            pd.read_csv(threshold_path) for threshold_path in threshold_filepaths
        ]
        for i in range(0, len(system_ids)):
            threshold_data_frames[i]["system_id"] = system_ids[i]
            threshold_data_frames[i] = pd.melt(threshold_data_frames[i], id_vars="system_id")
        sys_list = [aegis.acteval.system.System(sys_id, sys_df, thresh_df)
                    for (sys_id, sys_df, thresh_df)
                    in zip(system_ids, system_data_frames, threshold_data_frames)]

        return sys_list
