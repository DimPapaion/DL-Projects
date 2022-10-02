from setuptools import setup

setup(
    name='DRL-GAME-ATARI-BREAKOUT',
    version='1.0',
    packages=['pip install gym pyvirtualdisplay > /dev/null 2>&1',
              'apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1',
              'apt-get update > /dev/null 2>&1',
              'apt-get install cmake > /dev/null 2>&1',
              'pip install --upgrade setuptools 2>&1',
              'pip install ez_setup > /dev/null 2>&1',
              'pip install gym[atari] > /dev/null 2>&1',
              'pip install gym-retro',
              'pip install colabgymrender',
              '!unrar x Roms.rar',
              '!unzip ROMS.zip',
              '!wget http://www.atarimania.com/roms/Roms.rar',
              '!python -m atari_py.import_roms "./ROMS"',
              'apt-get install -y xvfb x11-utils',
             ' pip install gym[box2d]==0.17.* pyvirtualdisplay==0.2.* PyOpenGL==3.1.* PyOpenGL-accelerate==3.1.*'],
    url='-',
    license='-',

    description='Breakout Atari'
)
