conda activate jupyterbook
jupyter-book build --html
python3 src/inject_colab_toolbar.py
ghp-import -n -p -f _build/html
