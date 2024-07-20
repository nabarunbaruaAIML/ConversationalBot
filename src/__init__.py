from loguru import logger
logger.add("log/log.log",rotation="86400 seconds",retention=24)