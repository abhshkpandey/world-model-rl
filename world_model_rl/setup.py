from setuptools import setup, find_packages

setup(
    name="world_model_rl",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "transformers",
        "tensorboard",
        "matplotlib",
        "tqdm",
        "scipy",
        "gym",
        "stable-baselines3",
        "rich"
    ],
    entry_points={
        "console_scripts": [
            "train = scripts.train:main",
            "test = scripts.test:main",
            "inference = scripts.inference:main"
        ]
    },
    author="Viora AI labs",
    description="An advanced World Model RL framework",
    license="MIT"
)
