from setuptools import setup

package_name = 'imitrob_hri'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/,/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='petr',
    maintainer_email='petr.vanc@cvut.cz',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "mm_node = imitrob_hri.merging_modalities.modality_merger_node:main",
            #'listener = my_great_rostwo_package.sub:main',
        ],
    },
)
