"""Install Compacter."""
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

description = "PyTorch CUDA kernel implementation of intrinsic dimension operation."


def setup_package():
    ext_modules = []
    if torch.cuda.is_available():
        ext_modules = [
            CUDAExtension(
                "intrinsic.fwh_cuda",
                sources=[
                    "intrinsic/fwh_cuda/fwh_cpp.cpp",
                    "intrinsic/fwh_cuda/fwh_cu.cu",
                ],
            )
        ]

    setuptools.setup(
        name="intrinsic",
        version="0.0.1",
        description=description,
        long_description=description,
        long_description_content_type="text/markdown",
        author="Rabeeh Karimi Mahabadi",
        license="MIT License",
        packages=setuptools.find_packages(
            exclude=["docs", "tests", "scripts", "examples"]
        ),
        dependency_links=[
            "https://download.pytorch.org/whl/torch_stable.html",
        ],
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9.7",
        ],
        keywords="text nlp machinelearning",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        install_requires=[],
    )


if __name__ == "__main__":
    setup_package()
