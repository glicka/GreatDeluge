#!/usr/bin/env python

from distutils.core import setup

setup(name='GreatDeluge',
      version='0.0.1',
      description='Time optimization mathematical simulation utilizing the Great Deluge hypothesis.',
      author='Adam Glick',
      author_email='glicka@proton.me',
      url='https://github.com/glicka/GreatDeluge',
      packages=['greatdeluge'],
      package_dir={'greatdeluge': 'greatdeluge'},
      license="Refer to LICENSE file and CITATION.cff"
     )