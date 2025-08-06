import sys
import logging
import pickle

MINIMAL_LOGGING = False
LOGGING_FILE = "EMCrafter.log"

class Base:
    def __init__(self, verbose=1):
        """
        Initialize the Base class.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level for printing messages. Defaults to 1.
        """
        self.v = verbose
        self.class_name = self.__class__.__name__
        self._setup_logging()
        if verbose > 1:
            self.logger.info(f"{self.__class__.__name__} initialized")
    
    def _setup_logging(self):
        """
        Set up logging for this class.

        Sets up a logger with a format specified by 'logger_format'. The logger
        is set to log to both stdout and a file. The level of the logger is set
        to 'logging.INFO'.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger_format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        if MINIMAL_LOGGING:
            logger_format = '%(message)s'

        root_logger = logging.getLogger()

        if not root_logger.handlers:
            formatter = logging.Formatter(logger_format)

            # Stream to stdout
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)

            # Also log to a file
            file_handler = logging.FileHandler(LOGGING_FILE)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            root_logger.setLevel(logging.INFO)
    
    @property
    def logger(self):
        """
        A logger object with the name of the class.

        Returns
        -------
        logging.Logger
            A logger object with the name of the class.
        """
        return logging.getLogger(self.__class__.__name__)
    
    def save(self, file):
        """
        Save the current state of the object to a file using pickle.

        Parameters
        ----------
        file : str
            The file path where the object's state will be saved.

        Returns
        -------
        None
        """
        if self.v > 1: self.logger.info(f"Saving {self.class_name} object  to  {file}")
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def load(self, file, verbose=None):
        """
        Loads a saved object from a file using pickle.

        Parameters
        ----------
        file : str
            The file path from which the object's state was saved.

        Returns
        -------
        self : object
            The loaded object.
        """
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, self.__class__):
            raise TypeError(f"Loaded object is not a {self.class_name} object.")

        if verbose is not None: obj.v = verbose
        if obj.v > 1: obj.logger.info(f"Loaded {self.class_name} object from {file}")
        return obj

def _set_logging(minimal=MINIMAL_LOGGING, file_path=LOGGING_FILE):
    logger_format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    if minimal:
        logger_format = '%(message)s'

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if not root_logger.handlers:
        formatter = logging.Formatter(logger_format)

        # Stream to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        # Also log to a file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.setLevel(logging.INFO)
