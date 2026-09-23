from rl_games.core.adapter import Adapter


def make_adapter(config: dict) -> Adapter:
    """Build the adapter named by ``config['adapter']``."""
    name = config["adapter"]
    if name == "browser":
        from rl_games.adapters.browser import BrowserAdapter
        return BrowserAdapter(config)
    raise ValueError(f"unknown adapter {name!r}")
