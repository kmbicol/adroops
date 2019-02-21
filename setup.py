from setuptools import setup

setup(name='adroops',
      version='0.0.1',
      description='Filter Stabilization of Advection-Dominated Advection-Diffusion-Reaction Problems',
      url='http://github.com/kmbicol/adroops',
      author='Kayla M Bicol',
      author_email='kaylabicol@gmail.com',
      license='MIT',
      packages=['adroops'],
      include_package_data=True,
      install_requires=[
                        'matplotlib',
                        'numpy',
                        ],
      keywords=['bicol'],
      zip_safe=False)
