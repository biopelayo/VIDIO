from setuptools import setup, find_packages

setup(
    name='vidio',
    version='0.1.0',
    description='VIDIO - Vision-Integrated Diagnostic Imaging Orchestrator',
    author='VIDIO Team',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'falcon>=3.1',
        'SQLAlchemy>=2.0',
        'psycopg2-binary>=2.9',
        'PyJWT>=2.8',
        'bcrypt>=4.1',
        'opencv-python>=4.9',
        'numpy>=1.26',
        'scipy>=1.13',
        'pydicom>=2.4',
        'SimpleITK>=2.3',
        'nibabel>=5.2',
        'torch>=2.2',
        'monai>=1.3',
        'scanpy>=1.10',
        'squidpy>=1.4',
    ],
    entry_points={
        'console_scripts': [
            'vidio=VidioTool:main',
        ],
    },
)
