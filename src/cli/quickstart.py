#!/usr/bin/env python3
"""
Quick Start Guide for Food-10 Image Classification Notebook
Run this script to verify setup and launch the notebook
"""

import subprocess
import sys
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required packages are installed"""
    print_section("📦 Checking Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'streamlit': 'Streamlit'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name:20} installed")
        except ImportError:
            print(f"❌ {package_name:20} MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install torch torchvision scikit-learn matplotlib seaborn pillow pandas numpy streamlit")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_notebook():
    """Check if notebook file exists"""
    print_section("📓 Checking Notebook File")
    
    notebook_path = Path("notebooks/Food_Classification_Complete_Project.ipynb")
    
    if notebook_path.exists():
        size_mb = notebook_path.stat().st_size / (1024*1024)
        print(f"✅ Notebook found: {notebook_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Notebook not found: {notebook_path}")
        return False

def check_dataset():
    """Check if dataset directory exists"""
    print_section("📁 Checking Dataset")
    
    dataset_path = Path("classification_dataset")
    
    if dataset_path.exists():
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"
        
        if train_path.exists() and test_path.exists():
            print(f"✅ Dataset directory found")
            print(f"   Location: {dataset_path}")
            
            # Count images
            train_images = list(train_path.rglob("*.jpg")) + list(train_path.rglob("*.png"))
            test_images = list(test_path.rglob("*.jpg")) + list(test_path.rglob("*.png"))
            
            print(f"   Training images: {len(train_images)}")
            print(f"   Test images: {len(test_images)}")
            return True
    
    print(f"⚠️  Dataset directory not found: {dataset_path}")
    print("   Dataset will be downloaded on first notebook run")
    return False

def launch_notebook():
    """Launch the Jupyter notebook"""
    print_section("🚀 Launching Notebook")
    
    notebook_path = Path("notebooks/Food_Classification_Complete_Project.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print(f"📓 Opening notebook: {notebook_path}\n")
    
    # Try different notebook launchers
    commands = [
        ["jupyter", "notebook", str(notebook_path)],
        ["jupyter-notebook", str(notebook_path)],
        ["python", "-m", "jupyter", "notebook", str(notebook_path)],
    ]
    
    for cmd in commands:
        try:
            print(f"Attempting: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    print("\n❌ Could not launch Jupyter notebook")
    print("Try manually with: jupyter notebook notebooks/Food_Classification_Complete_Project.ipynb")
    return False

def launch_streamlit():
    """Launch the Streamlit app"""
    print_section("🚀 Launching Streamlit App")
    
    app_path = Path("app.py")
    
    if not app_path.exists():
        print(f"⚠️  Streamlit app not found: {app_path}")
        print("\n📋 To create the app:")
        print("1. Run the notebook (Cell 11)")
        print("2. Copy the Streamlit code")
        print("3. Save as app.py")
        return False
    
    print(f"🎬 Launching Streamlit app\n")
    
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
        return True
    except FileNotFoundError:
        print("❌ Streamlit not found")
        print("Install with: pip install streamlit")
        return False

def show_menu():
    """Show main menu"""
    print_section("🍽️  Food-10 Image Classification - Quick Start")
    
    print("""
Choose an option:
  1. Check dependencies
  2. Check all setup
  3. Launch notebook
  4. Launch Streamlit app
  5. Show documentation
  6. Exit
""")

def show_documentation():
    """Show documentation info"""
    print_section("📚 Documentation Files")
    
    docs = {
        "NOTEBOOK_GUIDE.md": "Complete guide for using the notebook",
        "PRESENTATION_SUMMARY.md": "Project overview and summary",
        "NOTEBOOK_STRUCTURE.md": "Detailed notebook structure",
        "README.md": "Project README"
    }
    
    for filename, description in docs.items():
        path = Path(filename)
        if path.exists():
            print(f"✅ {filename:30} {description}")
        else:
            print(f"❌ {filename:30} {description}")
    
    print("\n📖 Open these files to learn more about the project!")

def main():
    """Main menu loop"""
    try:
        import torch
        import jupyter
        import streamlit
        
        while True:
            show_menu()
            
            try:
                choice = input("Enter your choice (1-6): ").strip()
                
                if choice == "1":
                    check_dependencies()
                
                elif choice == "2":
                    print_section("🔍 Running Full Setup Check")
                    
                    all_good = True
                    all_good = check_dependencies() and all_good
                    all_good = check_notebook() and all_good
                    check_dataset()  # Optional but nice to know
                    
                    if all_good:
                        print("\n✨ All systems ready! You can run the notebook now.")
                    else:
                        print("\n⚠️  Some checks failed. Please resolve issues above.")
                
                elif choice == "3":
                    if check_dependencies() and check_notebook():
                        launch_notebook()
                
                elif choice == "4":
                    if check_dependencies():
                        launch_streamlit()
                
                elif choice == "5":
                    show_documentation()
                
                elif choice == "6":
                    print("\n👋 Goodbye! Happy learning!")
                    sys.exit(0)
                
                else:
                    print("❌ Invalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except ImportError:
        print_section("⚠️  Quick Start Requires Key Packages")
        print("\nTo use this quick start tool, install:")
        print("pip install torch jupyter streamlit\n")
        print("Or manually run:")
        print("jupyter notebook notebooks/Food_Classification_Complete_Project.ipynb")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           🍕 FOOD-10 IMAGE CLASSIFICATION PROJECT 🍕          ║
║                                                                ║
║              Complete ML Pipeline + Deployment                ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\nPlease ensure you're in the project root directory:")
        print("cd Food_Image_Classification")
        print("\nOr run directly:")
        print("jupyter notebook notebooks/Food_Classification_Complete_Project.ipynb")
