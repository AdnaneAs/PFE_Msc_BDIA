"""
Dependency Scanner for PFE Backend
=================================

This script scans your backend code to find all imported libraries
and generates an accurate requirements.txt file.
"""

import os
import ast
import sys
import pkg_resources
from pathlib import Path
from collections import defaultdict
import importlib.util

def scan_python_files(directory):
    """Scan all Python files for imports"""
    imports = set()
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip common directories that don't contain our code
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse the AST to find imports
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
                                
                except Exception as e:
                    print(f"Warning: Could not parse {file_path}: {e}")
    
    print(f"📁 Scanned {len(python_files)} Python files")
    return imports

def get_installed_packages():
    """Get list of installed packages with versions"""
    installed = {}
    try:
        for dist in pkg_resources.working_set:
            installed[dist.project_name.lower()] = dist.version
    except Exception as e:
        print(f"Warning: Could not get installed packages: {e}")
    
    return installed

def map_import_to_package(import_name):
    """Map import names to actual package names"""
    # Common mappings where import name != package name
    mapping = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'dotenv': 'python-dotenv',
        'bs4': 'beautifulsoup4',
        'redis': 'redis',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
        'httpx': 'httpx',
        'aiofiles': 'aiofiles',
        'sentence_transformers': 'sentence-transformers',
        'transformers': 'transformers',
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'chromadb': 'chromadb',
        'langchain': 'langchain',
        'langgraph': 'langgraph',
        'openai': 'openai',
        'google': 'google-generativeai',
        'streamlit': 'streamlit',
        'nest_asyncio': 'nest_asyncio',
        'tiktoken': 'tiktoken',
        'plotly': 'plotly',
        'accelerate': 'accelerate',
        'pypdf': 'pypdf',
        'unstructured': 'unstructured',
        'pdf2image': 'pdf2image',
        'pytesseract': 'pytesseract',
        'requests': 'requests',
        'aiohttp': 'aiohttp',
        'multipart': 'python-multipart',
        'filelock': 'filelock',
        'cachetools': 'cachetools',
        'aiocache': 'aiocache'
    }
    
    return mapping.get(import_name.lower(), import_name)

def generate_requirements(backend_dir):
    """Generate requirements.txt based on actual imports"""
    print(f"🔍 Scanning {backend_dir} for dependencies...")
    
    # Scan for imports
    imports = scan_python_files(backend_dir)
    
    # Filter out standard library modules
    stdlib_modules = {
        'os', 'sys', 'json', 'logging', 'datetime', 'time', 'uuid', 'pathlib',
        'typing', 'asyncio', 'threading', 'multiprocessing', 'concurrent',
        'collections', 'functools', 'itertools', 'tempfile', 'shutil', 'glob',
        'io', 're', 'math', 'random', 'pickle', 'base64', 'hashlib', 'hmac',
        'urllib', 'http', 'email', 'xml', 'html', 'csv', 'sqlite3', 'dbm',
        'gzip', 'zipfile', 'tarfile', 'configparser', 'argparse', 'getpass',
        'platform', 'socket', 'ssl', 'subprocess', 'signal', 'atexit',
        'warnings', 'traceback', 'inspect', 'ast', 'dis', 'importlib'
    }
    
    # Get installed packages
    installed = get_installed_packages()
    
    # Build requirements
    requirements = []
    found_packages = set()
    
    print(f"📦 Found {len(imports)} imports:")
    for imp in sorted(imports):
        if imp in stdlib_modules:
            continue
            
        package_name = map_import_to_package(imp)
        
        # Check if package is installed
        if package_name.lower() in installed:
            version = installed[package_name.lower()]
            requirements.append(f"{package_name}=={version}")
            found_packages.add(package_name)
            print(f"  ✅ {imp} -> {package_name}=={version}")
        elif imp.lower() in installed:
            version = installed[imp.lower()]
            requirements.append(f"{imp}=={version}")
            found_packages.add(imp)
            print(f"  ✅ {imp}=={version}")
        else:
            # Try to guess version or use latest
            requirements.append(f"{package_name}")
            found_packages.add(package_name)
            print(f"  ❓ {imp} -> {package_name} (version unknown)")
    
    return sorted(requirements), found_packages

def main():
    """Main function"""
    print("🔍 PFE Backend Dependency Scanner")
    print("=================================")
    
    # Get backend directory
    backend_dir = "backend"
    if not os.path.exists(backend_dir):
        print(f"❌ Backend directory '{backend_dir}' not found!")
        print("Please run this script from the PFE_sys root directory")
        return
    
    # Generate requirements
    requirements, found_packages = generate_requirements(backend_dir)
    
    # Write to file
    output_file = "requirements_scanned.txt"
    with open(output_file, 'w') as f:
        f.write("# Auto-generated requirements based on actual imports\n")
        f.write("# Generated by scan_dependencies.py\n\n")
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"\n📝 Generated {output_file} with {len(requirements)} packages")
    print(f"💾 Found packages: {', '.join(sorted(found_packages))}")
    
    # Compare with existing requirements.txt
    existing_req_file = os.path.join(backend_dir, "requirements.txt")
    if os.path.exists(existing_req_file):
        print(f"\n🔄 Comparing with existing {existing_req_file}...")
        
        with open(existing_req_file, 'r') as f:
            existing_lines = [line.strip() for line in f.readlines() 
                            if line.strip() and not line.startswith('#')]
        
        existing_packages = set()
        for line in existing_lines:
            if '==' in line:
                package = line.split('==')[0]
            elif '>=' in line:
                package = line.split('>=')[0]
            else:
                package = line
            existing_packages.add(package.strip())
        
        new_packages = found_packages - existing_packages
        missing_packages = existing_packages - found_packages
        
        if new_packages:
            print(f"➕ New packages found: {', '.join(new_packages)}")
        
        if missing_packages:
            print(f"➖ Existing packages not used: {', '.join(missing_packages)}")
    
    print(f"\n✅ Scan complete! Check {output_file} for the results.")

if __name__ == "__main__":
    main()
