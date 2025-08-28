LANGUAGES = [
    "English",
    "Polish",
    "German",
    "Dutch",
    "Spanish",
    "Italian",
    "Other"
]

MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5"
]

DEFAULT_PARALLEL_REQUESTS = 3

DEFAULT_PRICE_PER_1000_TOKENS = 0.001

# Connection pool configuration
CONNECTION_POOL_CONFIG = {
    "num_pools": 15,
    "maxsize": 30,
    "connect_timeout": 2.0,
    "read_timeout": 15.0
}
