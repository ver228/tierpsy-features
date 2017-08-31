#!/usr/bin/env python

from setuptools import setup

MODULE_NAME = "tierpsy_features"
AUTHOR = 'Avelino Javer'
AUTHOR_EMAIL = 'avelino.javer@imperial.ac.uk'
URL = 'https://github.com/ver228/tierpsy-features'
DOWNLOAD_URL = 'https://github.com/ver228/tierpsy-features'
DESCRIPTION = "tierpsy_features: C. elegans behaviour features."
exec(open(MODULE_NAME + '/_version.py').read())
VERSION = __version__


#install setup
setup(name=MODULE_NAME,
   version=VERSION,
   description=DESCRIPTION,
   author=AUTHOR,
   author_email=AUTHOR_EMAIL,
   url=URL,
   packages=['tierpsy_features'],
   package_data =  {'tierpsy_features': ['extras/*']},
   include_package_data=True
   )
