Demo PCA algorithm
==================

## Environment setup ##

Install virtualenv

```bash
sudo apt install python-pip
pip install virtualenvwrapper
```

Add two lines to ~/.bashrc (~/.profile)
```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel

# load virtualenvwrapper for python (after custom PATHs)
venvwrap="virtualenvwrapper.sh"
/usr/bin/which $venvwrap
if [ $? -eq 0 ]; then
    venvwrap=`/usr/bin/which $venvwrap`
    source $venvwrap
fi
```

Run script
```bash
. ~/.local/bin/virtualenvwrapper.sh
```

Create virtual environment for matching app
```bash
mkvirtualenv -p python3.5 pca
```

## Install dependencies ##
```bash
pip install -r requirements.txt
```

## Install library ##
```bash
pip install pcalib
```

## Run tests ##

```bash
pytest tests
```

## Run app ##

```bash
python demo.py
```