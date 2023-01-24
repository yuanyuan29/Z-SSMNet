import setuptools

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        version='1.0.0',
        author_email='yyua9990@uni.sydney.edu.au',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/yuanyuan29/Z-SSMNet',
        project_urls={
            "Bug Tracker": "https://github.com/yuanyuan29/Z-SSMNet/issues"
        },
        license='Apache 2.0',
        packages=setuptools.find_packages('src', exclude=['tests', 'inference']),
        package_data={'': [
            'splits/*/*.json',
        ]},
        include_package_data=True,
    )
