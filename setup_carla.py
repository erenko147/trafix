"""
CARLA Python API Kurulum Scripti
=================================
CARLA 0.9.16 wheel'ini mevcut Python versiyonu ile uyumlu hale getirir.
Wheel cp312 ile derlenmis olsa bile Python 3.13 altinda calistirir.

Kullanim:
    python setup_carla.py
    python setup_carla.py --carla-path "D:/Baska/CARLA_0.9.16"
"""

import argparse
import os
import shutil
import site
import sys
import zipfile

DEFAULT_CARLA_PATH = r"C:\Users\Furkan\Downloads\CARLA_0.9.16"


def find_wheel(carla_root: str) -> str:
    dist_dir = os.path.join(carla_root, "PythonAPI", "carla", "dist")
    if not os.path.exists(dist_dir):
        sys.exit(f"HATA: Dizin bulunamadi: {dist_dir}")
    wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl") and "carla" in f]
    if not wheels:
        sys.exit(f"HATA: {dist_dir} icinde .whl dosyasi bulunamadi.")
    return os.path.join(dist_dir, sorted(wheels)[-1])   # en son wheel


def get_site_packages() -> str:
    """Aktif venv'in site-packages dizinini bul."""
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    for p in paths:
        if "site-packages" in p and os.path.isdir(p):
            return p
    # Fallback: sys.path icinde ara
    for p in sys.path:
        if "site-packages" in p and os.path.isdir(p):
            return p
    sys.exit("HATA: site-packages dizini bulunamadi.")


def install_carla_wheel(wheel_path: str, site_packages: str):
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"[KURULUM] Python versiyonu: {py_ver}")
    print(f"[KURULUM] Wheel: {wheel_path}")
    print(f"[KURULUM] Hedef: {site_packages}")

    carla_pkg_dir = os.path.join(site_packages, "carla")

    # Onceki kurulumu temizle
    if os.path.exists(carla_pkg_dir):
        print(f"[KURULUM] Onceki kurulum siliniyor: {carla_pkg_dir}")
        shutil.rmtree(carla_pkg_dir)

    # Dist-info temizle
    for entry in os.listdir(site_packages):
        if entry.startswith("carla-") and entry.endswith(".dist-info"):
            shutil.rmtree(os.path.join(site_packages, entry))

    # Wheel'i ac ve icerigi site-packages'a kopyala
    print(f"[KURULUM] Wheel extract ediliyor...")
    with zipfile.ZipFile(wheel_path, "r") as z:
        z.extractall(site_packages)

    # .pyd dosyasini mevcut Python versiyonuna gore yeniden adlandir
    pyd_files = [
        f for f in os.listdir(carla_pkg_dir)
        if f.endswith(".pyd") and "libcarla" in f
    ]
    if not pyd_files:
        print("[UYARI] libcarla.pyd dosyasi wheel icinde bulunamadi.")
        return

    for pyd_file in pyd_files:
        old_path = os.path.join(carla_pkg_dir, pyd_file)
        # Hedef ad: libcarla.<pyver>-win_amd64.pyd
        new_name = f"libcarla.{py_ver}-win_amd64.pyd"
        new_path = os.path.join(carla_pkg_dir, new_name)

        if pyd_file == new_name:
            print(f"[KURULUM] .pyd zaten dogru isimde: {new_name}")
        else:
            # Hem orijinal hem de yeni isimle tut (import basarisizliga karsi)
            shutil.copy2(old_path, new_path)
            print(f"[KURULUM] {pyd_file} -> {new_name} (kopya olusturuldu)")

    print("\n[KURULUM] CARLA Python API kuruldu.")
    print("[DOGRULAMA] Test ediliyor...")
    _verify()


def _verify():
    try:
        import importlib
        import importlib.util

        # site-packages'tan taze import
        if "carla" in sys.modules:
            del sys.modules["carla"]
        import carla
        print(f"[OK] carla modulu yuklendi. Versiyon: {carla.__version__ if hasattr(carla, '__version__') else 'N/A'}")
        # Temel sinif kontrolu
        _ = carla.Location(x=0, y=0, z=0)
        print("[OK] carla.Location() calisiyor.")
        _ = carla.Transform()
        print("[OK] carla.Transform() calisiyor.")
        print("\n[BASARILI] CARLA Python API hazir!")
    except ImportError as e:
        print(f"\n[HATA] Import basarisiz: {e}")
        print("\nOlasi neden: libcarla.pyd Python 3.12 icin derlenmis, 3.13 ile ABI uyumsuzlugu.")
        print("Cozum: Python 3.12 kurun ve .venv'i yeniden olusturun:")
        print("  1. https://www.python.org/downloads/release/python-3129/ adresinden Python 3.12'yi indirin")
        print("  2. py -3.12 -m venv .venv")
        print("  3. .venv\\Scripts\\activate && pip install -r requirements.txt")
        print(f"  4. pip install {sys.argv[0].replace('setup_carla.py','') + '...'}")
        sys.exit(1)
    except Exception as e:
        print(f"[HATA] Beklenmeyen hata: {e}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="CARLA Python API kurulumu")
    p.add_argument(
        "--carla-path", default=DEFAULT_CARLA_PATH,
        help=f"CARLA kurulum dizini (varsayilan: {DEFAULT_CARLA_PATH})"
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="Sadece mevcut kurulumu dogrula, yeniden kurma"
    )
    args = p.parse_args()

    if args.verify_only:
        _verify()
        return

    wheel = find_wheel(args.carla_path)
    site_pkg = get_site_packages()
    install_carla_wheel(wheel, site_pkg)


if __name__ == "__main__":
    main()
