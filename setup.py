from setuptools import setup

if __name__ == "__main__":
    setup(
        # setuptools_scm
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        # package metadata
        name="dea-burn-cube",
        packages=["dea_burn_cube"],
        python_requires=">=3.8",
        install_requires=[
            "click",
            "pytest",
            "coverage",
            "geopandas",
            "datacube",
            "boto3",
            "botocore",
            "pyarrow",
            "fsspec",
            "s3fs",
            "moto[s3]",
            "tqdm",
            "dea-tools",
            "aiohttp",
        ],
        entry_points={
            "console_scripts": ["dea-burn-cube=dea_burn_cube.__main__:main"],
        },
    )