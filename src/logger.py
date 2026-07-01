import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    "Configured Logger"
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    
    #Console Handler
    console_handler = logging.StreamHandler()   
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #FileHandler
    log_file = Path(__file__).parent.parent / "logs" / "chatbot.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.error("This is an error message")

