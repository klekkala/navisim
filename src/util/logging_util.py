import logging
import datetime
import os

def logging_setup(level=logging.INFO):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("log", exist_ok=True)
    log_filename = f"log/{current_time}.log"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(message)s',
        level=level
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # This line adds console output as well