from setuptools import setup

setup(name='Fluorify',
      version='0.1',
      description='Automated Fluorine Scanning',
      url='https://github.com/adw62/Fluorify',
      author='AW',
      author_email='None',
      license='None',
      packages=['Fluorify'],
      entry_points = {'console_scripts':['Fluorify = Fluorify.cli:main']})
