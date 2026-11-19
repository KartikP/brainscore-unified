"""Package prebuilt frontend assets without invoking Node on user machines."""
from pathlib import Path
import runpy

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def check_frontend():
    module = Path(__file__).resolve().parent / "backend/umi_lab/frontend.py"
    runpy.run_path(str(module))["frontend_directory"]()


class BuildPy(build_py):
    def run(self):
        check_frontend()
        super().run()


class Sdist(sdist):
    def run(self):
        check_frontend()
        super().run()


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
