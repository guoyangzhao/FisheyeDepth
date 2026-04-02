""" Add package to PYTHONPATH
"""
import sys
import os
import logging
import coloredlogs

package_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, package_path)

def manage_package_logging():
    coloredlogs.install(logging.CRITICAL)