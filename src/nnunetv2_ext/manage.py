import importlib.util
import sys
import shutil
import argparse
from pathlib import Path

def get_paths():
    """Gets the source and destination paths for the extension files."""
    # Source: our own package's source files
    ext_project_root = Path(__file__).resolve().parent.parent
    ext_src_root = ext_project_root / "src" / "nnunetv2"
    if not ext_src_root.is_dir():
        raise FileNotFoundError(f"Extension source directory not found at '{ext_src_root}'")

    # Destination: the installed nnunetv2 library
    try:
        spec = importlib.util.find_spec("nnunetv2")
        if spec is None or spec.origin is None:
            raise ImportError
        nnunet_dest_root = Path(spec.origin).parent
    except ImportError:
        raise ImportError("Could not find installed 'nnunetv2' library. Is it installed in this environment?")

    return ext_src_root, nnunet_dest_root

def install(use_copy: bool):
    """Installs extension files into the nnunetv2 site-packages directory."""
    print("--- Installing ventriscar-nnunet-ext into nnunetv2 ---")
    ext_src_root, nnunet_dest_root = get_paths()
    print(f"Found extensions source: {ext_src_root}")
    print(f"Found nnunetv2 installation: {nnunet_dest_root}")

    mappings = [("nets", "nets"), ("training/nnUNetTrainer", "training/nnUNetTrainer")]
    installed_count = 0

    for src_subdir, dest_subdir in mappings:
        src_path = ext_src_root / src_subdir
        dest_path = nnunet_dest_root / dest_subdir
        if not dest_path.is_dir():
            print(f"WARNING: Destination directory '{dest_path}' does not exist. Skipping.")
            continue

        for src_file in src_path.glob("*.py"):
            if src_file.name == "__init__.py":
                continue
            
            dest_file = dest_path / src_file.name
            # Remove existing file/symlink to ensure a clean installation
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()

            if use_copy:
                print(f"  -> Copying {src_file.name}...")
                shutil.copy2(src_file, dest_file)
            else:
                print(f"  -> Symlinking {src_file.name}...")
                dest_file.symlink_to(src_file)
            installed_count += 1
            
    if installed_count > 0:
        print(f"\nSUCCESS: {'Copied' if use_copy else 'Symlinked'} {installed_count} extension files.")
    else:
        print("\nWARNING: No extension files were installed.")

def verify():
    """Verifies that the custom trainer is discoverable by nnunetv2."""
    print("--- Verifying ventriscar-nnunet-ext installation ---")
    try:
        # We need to make sure the paths are correct for nnunetv2's utility
        spec = importlib.util.find_spec("nnunetv2.training.nnUNetTrainer")
        if not spec or not spec.submodule_search_locations:
            raise ImportError("Cannot find nnunetv2.training.nnUNetTrainer module location.")
        
        search_path = spec.submodule_search_locations
        
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        
        print("Searching for 'nnUNetTrainerUxLSTMEnc'...")
        trainer_class = recursive_find_python_class(search_path, 'nnUNetTrainerUxLSTMEnc', 'nnunetv2.training.nnUNetTrainer')
        
        if trainer_class is None:
             raise RuntimeError("Trainer class not found by nnU-Net's utility.")

        print(f"SUCCESS: Found trainer class: {trainer_class}")
        return True
    except Exception as e:
        print(f"ERROR: Verification failed. The custom extensions are not discoverable.")
        print(f"Details: {e}")
        return False

def main(args):

    try:
        if args.command == "install":
            install(args.copy)
            verify() # Automatically verify after install
        elif args.command == "verify":
            if not verify():
                sys.exit(1)
    except (FileNotFoundError, ImportError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage and verify the ventriscar-nnunet-ext installation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Install command
    parser_install = subparsers.add_parser("install", help="Install extensions into the nnunetv2 library.")
    parser_install.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    
    # Verify command
    subparsers.add_parser("verify", help="Verify that the extensions are discoverable.")
    
    args = parser.parse_args()
    
    main(args)