import uvicorn
import logging
from config import config

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(
        f"Starting ProcessGen Model Server on {config.api_host}:{config.api_port}"
    )
    logger.info(f"Embedding Model: {config.embedding_model_name}")
    logger.info(f"Rerank Model: {config.rerank_model_name}")
    logger.info(f"Embedding Device: {config.embedding_device}")
    logger.info(f"Rerank Device: {config.rerank_device}")

    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
        access_log=True,
    )
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(
        f"Starting ProcessGen Model Server on {config.api_host}:{config.api_port}"
    )

    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
        access_log=True,
    )
