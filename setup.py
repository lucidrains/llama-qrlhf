from setuptools import setup, find_packages

setup(
  name = 'llama-qrlhf',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Experimental Q-RLHF applied to Language Modeling. Made compatible with Llama of course',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/llama-qrlhf',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'reinforcement learning with human feedback',
    'q learning',
  ],
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
