from setuptools import setup

setup(name='LigCharOpt',
      version='0.1',
      description='Automated ligand charge optimisation',
      url='https://github.com/adw62/Ligand_Charge_Optimiser',
      author='AW',
      author_email='None',
      license='None',
      packages=['LigCharOpt'],
      entry_points = {'console_scripts':['LigCharOpt = LigCharOpt.cli:main']})
