from setuptools import setup, find_packages

setup(
    name="detecting_phishing_urls",
    version="0.1.0",
    description="A data science project for detecting phishing URLs",
    author="Polina Polskaia",
    author_email="polskaia@bu.edu",
    url="https://github.com/polinacsv/detecting-phishing-URLs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.24,<1.25",
        "pandas>=2.0,<2.1",
        "scikit-learn>=1.1,<1.2",
        "matplotlib>=3.6,<3.7"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)