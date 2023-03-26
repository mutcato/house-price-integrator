from dotenv import dotenv_values
from logging.config import dictConfig
from logram.handlers import Telegram

config = dotenv_values(".env")
Telegram.token = config["TELEGRAM_TOKEN"]
Telegram.chat_id = config["CHAT_ID"]
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[%(levelname)s] [%(asctime)s] [%(module)s] [line:%(lineno)d] %(message)s"
        },
        "simple": {"format": "%(levelname)s %(message)s"},
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": "logs/log",
            "formatter": "verbose",
            "encoding": "UTF-8",
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "telegram": {
            "level": "WARNING",
            "class": "logram.handlers.Telegram",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": True},
        "offices": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        "messagereport": {"handlers": ["console", "telegram",  "file"], "level": "INFO", "propagate": True},
        "qinspect": {"handlers": ["console"], "level": "INFO", "propagate": True,},
    },
}
print("settings imported")
dictConfig(LOGGING)


