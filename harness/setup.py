from setuptools import find_packages, setup

setup(
    name="determined",
    version="0.19.12-dev0",
    author="Determined AI",
    author_email="hello@determined.ai",
    url="https://determined.ai/",
    description="Determined Deep Learning Training Platform",
    long_description="See https://docs.determined.ai/ for more information.",
    license="Apache License 2.0",
    classifiers=["License :: OSI Approved :: Apache Software License"],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    # Technically, we haven't supported 3.6 or tested against it since it went EOL.  But some users
    # are still using it successfully so there's hardly a point in breaking them.
    python_requires=">=3.6",
    package_data={"determined": ["py.typed"]},
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "packaging",
        "numpy>=1.16.2",
        "psutil",
        "pyzmq>=18.1.0",
        "yogadl==0.1.4",
        # Common:
        "backoff",
        "certifi",
        "filelock",
        "google-cloud-storage",
        "hdfs>=2.2.2",
        "lomond>=0.3.3",
        "pathspec>=0.6.0",
        "azure-core",
        "azure-storage-blob",
        "termcolor>=1.1.0",
        "boto3",
        # CLI:
        "argcomplete>=1.9.4",
        "gitpython>=3.1.3",
        "pyOpenSSL>= 19.1.0",
        "python-dateutil",
        "pytz",
        "tabulate>=0.8.3",
        # det preview-search "pretty-dumps" a sub-yaml with an API added in 0.15.29
        "ruamel.yaml>=0.15.29",
        # Deploy
        "docker[ssh]>=3.7.3",
        "google-api-python-client>=1.12.1",
        "paramiko>=2.4.2",  # explicitly pull in paramiko to prevent DistributionNotFound error
        "docker-compose>=1.13.0",
        "tqdm",
        "appdirs",
        # docker-compose has a requirement not properly propagated with semi-old pip installations;
        # so we expose that requirement here.
        "websocket-client<1",
        # Telemetry
        "analytics-python",
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "det = determined.cli.__main__:main",
            "det-deploy = determined.deploy.__main__:main",
        ]
    },
)
