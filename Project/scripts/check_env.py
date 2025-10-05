#!/usr/bin/env python

"""
Quick environment check for project dependencies.
Does not install anything; only reports versions and availability.
"""

import sys
import importlib


def try_import(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        return None


def main():
    print("== Environment Check ==")
    print(f"Python: {sys.version.split()[0]}")

    torch = try_import("torch")
    if torch is None:
        print("torch: MISSING")
    else:
        cuda = torch.version.cuda
        print(f"torch: {torch.__version__} (CUDA {cuda or 'CPU'})")
        print(f"CUDA available: {torch.cuda.is_available()}")

    for pkg in ["torchvision", "mmengine", "mmcv", "mmdet", "mmpose"]:
        mod = try_import(pkg)
        if mod is None:
            print(f"{pkg}: MISSING")
        else:
            ver = getattr(mod, "__version__", "unknown")
            print(f"{pkg}: {ver}")

    for pkg in ["numpy", "opencv", "cv2", "albumentations", "pycocotools"]:
        modname = pkg
        if pkg == "opencv":
            modname = "cv2"
        mod = try_import(modname)
        if mod is None:
            print(f"{pkg}: MISSING")
        else:
            ver = getattr(mod, "__version__", "unknown")
            print(f"{pkg}: {ver}")

    print("== Done ==")


if __name__ == "__main__":
    main()
