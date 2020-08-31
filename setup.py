from setuptools import setup

with open("requirements.txt") as f_req:
    required_list = [line.rstrip() for line in f_req.readlines()]

setup(
    name='MCR_Constraints',
    version='0.1',
    package=['mcr_const'],
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Xiaohui Qu',
    author_email='xiaqu@bnl.gov',
    description='Constraints to Embed Prior Knowledge in MCr',
    python_requires='>=3.7',
    install_requires=required_list,
    entry_points={
        "console_scripts": []
    }
)
