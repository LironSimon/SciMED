# library import
import os
import re
import json
from glob import glob


# project import


class TPOTresultsExtractor:
    """
    This class is responsible to get a file or folder with files produced by the
    MultiTPOTrunner library with pipeline configurations and extract only the description of the pipeline itself
    """

    # CONSTS #
    OPEN_DIVIDER = "make_pipeline("
    CLOSE_DIVIDER = "exported_pipeline."
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def process_folder(folder_path: str,
                       answer_path: str = None):
        """
        :param folder_path: path to a folder with MultiTPOTrunner produced <>.py files
        :param answer_path: path to the results, if None, just return the string with the answer
        """
        answer = [TPOTresultsExtractor.process_file(file_path=file_path,
                                                    answer_path=None)
                  for file_path in glob(os.path.join(folder_path, "*.py"))]
        if answer_path is not None:
            with open(answer_path, "w") as answer_file:
                json.dump(answer, answer_file)
        else:
            return answer

    @staticmethod
    def process_file(file_path: str,
                     answer_path: str = None):
        """
        :param file_path: path to the MultiTPOTrunner produced <>.py file
        :param answer_path: path to the results, if None, just return the string with the answer
        """
        data_text = ""
        with open(file_path, "r") as input_file:
            data_text = input_file.read()
        open_index = data_text.find(TPOTresultsExtractor.OPEN_DIVIDER)
        close_index = data_text.find(TPOTresultsExtractor.CLOSE_DIVIDER)
        pipeline = data_text[open_index + len(TPOTresultsExtractor.OPEN_DIVIDER):close_index].strip().replace("\n", "")
        while ", " in pipeline:
            pipeline = pipeline.replace(", ", ",")
        if answer_path is not None:
            with open(answer_path, "w") as result_file:
                result_file.write(pipeline)
        else:
            return pipeline
