import subprocess
import re

def install_default_version(package_name):
    # Get the default version of the package from PyPI
    cmd = ['pip', 'search', package_name]
    search_output = subprocess.check_output(cmd, universal_newlines=True)
    default_version_match = re.search(r'\(([^)]+)\)', search_output)
    if default_version_match:
        default_version = default_version_match.group(1)
        # Install the default version using pip
        subprocess.run(['pip', 'install', f'{package_name}=={default_version}'])

with open('requirements.txt', 'r') as f:
    for line in f:
        package_spec = line.strip()
        # Extract package name and version from the line
        package_name, _, specified_version = package_spec.partition('==')
        try:
            # Attempt to install the specified version using pip
            subprocess.run(['pip', 'install', package_spec], check=True)
        except subprocess.CalledProcessError:
            # If installation fails, install the default version
            install_default_version(package_name)

