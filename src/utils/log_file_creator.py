"""
    Generative_AI_Authorship_Verification Project:
        utils:
            log_file_creator.py

"""

# ============================ Third Party libs =======================
import os
import re


class CreateLogFile:
    """A class for managing log files and versions."""
    def __init__(self, base_folder_path: str):
        """
        Initialize the CreateLogFile object.

        Args:
            base_folder_path (str): The base folder path where log files are stored.
        """
        self.base_folder_path = base_folder_path
        os.makedirs(self.base_folder_path, exist_ok=True)

    def find_folders(self) -> list:
        """
        Find all folders within the base folder path.

        Returns:
            list: A list of folder names.
        """
        folders = [f for f in os.listdir(self.base_folder_path) if
                   os.path.isdir(os.path.join(self.base_folder_path, f))]
        return folders

    def extract_version(self) -> int:
        """
        Extract the latest version number from the existing folders.

        Returns:
            int: The latest version number.
        """
        folders = self.find_folders()
        pattern = re.compile(r'version_\d+$')
        # Extract items matching the pattern
        folders = [item for item in folders if pattern.match(item)]
        if not folders:
            return -1
        sorted_versions = sorted(folders, key=lambda x: int(x.split('_')[1]))
        version = int(sorted_versions[-1].split("version_")[-1])
        return version

    def create_versioned_file(self) -> str:
        """
        Create a new versioned log file.

        Returns:
            str: The path to the newly created versioned folder.
        """
        version = self.extract_version()
        folder_name = f"version_{version+1}"
        folder_path = os.path.join(self.base_folder_path, folder_name)
        os.makedirs(folder_path)
        return folder_path
