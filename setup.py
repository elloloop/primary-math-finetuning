from setuptools import find_packages, setup

setup(
    name="qwen-math-finetuning",
    version="1.0.0",
    description="Fine-tuning Qwen2.5 for primary-level math word problems.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.46",
        "peft>=0.13",
        "accelerate>=0.34",
        "trl>=0.11",
        "datasets>=2.16",
        "tensorboard>=2.15",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "wandb": [
            "wandb>=0.16",
        ],
    },
)
