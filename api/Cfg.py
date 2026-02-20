"""
Global Configuration Singleton
==============================

This module holds the global application configuration as a module-level
singleton variable ``gCfg``. The configuration dictionary is loaded once
at application startup (typically from a JSON file) and made available to
every module in the ``api`` package via::

    from api import Cfg
    db_url = Cfg.gCfg['database']['url']

Attributes
----------
gCfg : dict or None
    The global configuration dictionary. ``None`` until the application
    bootstrap code populates it. Expected top-level keys include:

    - ``'database'`` -- database connection parameters.
    - ``'auth'`` -- authentication settings (``secret_key``, ``algorithm``,
      ``expiration_days``).
    - ``'repository'`` -- file-storage paths.
    - ``'pipelines'`` -- pipeline / analysis runner settings.

Notes
-----
Because ``gCfg`` is a plain module attribute, any module that imports
``Cfg`` obtains a reference to the *same* dictionary object, providing
a lightweight configuration-sharing mechanism without requiring
dependency injection.
"""

gCfg = None
