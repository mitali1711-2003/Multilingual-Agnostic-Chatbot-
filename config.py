import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'chatbot.db')}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    SECURITY_PASSWORD_SALT = os.environ.get("SECURITY_PASSWORD_SALT", "change-me-salt")


class DevConfig(Config):
    DEBUG = True


class ProdConfig(Config):
    DEBUG = False
