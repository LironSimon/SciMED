# library imports
import logging
from termcolor import colored

# project imports


class Logger:
    """
    Our project logger
    """

    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.setLevel(logging.DEBUG)

    def __init__(self,save_path):
        logging.basicConfig(filename=save_path,
                            format='%(asctime)s %(message)s',
                            filemode='w')

    @staticmethod
    def print(message: str):
        Logger.logger.info(message)
        print("Info: {}".format(message))

    @staticmethod
    def info(message: str):
        Logger.print(message=message)

    @staticmethod
    def important(message: str):
        Logger.logger.critical(message)
        print("Important: {}".format(colored(message, "bold")))

    @staticmethod
    def debug(message: str):
        Logger.logger.debug(message)
        print("Debug: {}".format(colored(message, "red")))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Logger>"
