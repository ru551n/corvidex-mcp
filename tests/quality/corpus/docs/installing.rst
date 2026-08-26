Installing the toolchain
========================

Install the toolchain with pip from the public PyPI index. The
recommended setup is a dedicated virtual environment:

.. code-block:: sh

    python -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip
    pip install my-toolchain

Python 3.10 or newer is required. On systems without a C compiler,
install the prebuilt platform wheel (pip selects it automatically).

To upgrade, re-run ``pip install --upgrade my-toolchain`` inside the
same virtual environment. To uninstall, use ``pip uninstall
my-toolchain``.