from setuptools import setup, find_packages

setup(  
    name = "deep-crf",
    version = "1.5.2",
    keywords = ("lstm-crf", "crf"),
    description = "Sequence label tool base on Bi-LSTM-CRF mode.",
    long_description = "Sequence label tool base on Bi-LSTM-CRF mode.",
    license = "GPLv3",
  
    url = "",
    author = "wangsihong",
    author_email = "wangsihong@live.com",
  
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires=[],
  
    entry_points = {
        'console_scripts': [
            'deepcrf_learn = deepcrf.deepcrf_learn:main',
            'deepcrf_eval = deepcrf.deepcrf_eval:main',
            'deepcrf_save = deepcrf.deepcrf_save:main',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    zip_safe=False
)
