# from importlib.metadata import entry_points
import os, glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
_ext_src_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guided_diffusion/clib")
_ext_sources = glob.glob("{}/*.cpp".format(_ext_src_root)) + glob.glob("{}/*.cu".format(_ext_src_root))
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))


setup(
    name="basic-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "mpi4py", "av"],
    ext_modules=[
        CUDAExtension(
            name='guided_diffusion.clib._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    entry_points={
        'console_scripts': [
            'launch-train  = neural_diffusion.launch:main_train',
            'launch-eval   = neural_diffusion.launch:main_eval',
            'launch-sample = neural_diffusion.launch:main_sample'
        ]
    }
)
