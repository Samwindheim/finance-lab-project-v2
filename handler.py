import json
import os
import sys

# Ensure the root and src directories are in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.main import extract_new_command, extract_historical_command
from src.logger import setup_logger
import src.config as config

logger = setup_logger(__name__)

def handler(event, context):
    """
    AWS Lambda entry point.
    'event' contains the data passed to the function.
    """
    try:
        # 1. Parse input from the event
        # Example event: {"command": "extract-new", "source_link": "https://mfn.se/..."}
        command = event.get("command")
        
        # Create a mock 'args' object that looks like what argparse produces
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.source_link = kwargs.get("source_link", None)
                self.extraction_field = kwargs.get("extraction_field", None)
                self.issue_id = kwargs.get("issue_id", None)
                self.output_dir = str(config.OUTPUT_JSON_DIR)
                self.index_dir = str(config.FAISS_INDEX_DIR)
                self.n = kwargs.get("n", 3) # for query command if needed
                self.yes = kwargs.get("yes", True) # for clear command if needed

        args = Args(**event)

        # 2. Route to the correct logic
        if command == "extract-new":
            extract_new_command(args)
        elif command == "extract-historical":
            extract_historical_command(args)
        else:
            return {"statusCode": 400, "body": f"Unknown command: {command}"}

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Extraction complete"})
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {"statusCode": 500, "body": str(e)}
