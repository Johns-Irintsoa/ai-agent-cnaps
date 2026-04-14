import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=os.environ["API_HOST"],
        port=int(os.environ["API_PORT"]),
        reload=False,
    )
