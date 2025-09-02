import setuptools

# 读取requirements.txt中的依赖项
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setuptools.setup(
    name="airsim-droneserver",
    version="1.0.0",
    author="Custom DroneServer",
    description="AirSim DroneServer - A server for controlling drones in AirSim simulator",
    long_description="A server implementation for controlling drones in the AirSim simulator, providing socket-based API for drone operations.",
    long_description_content_type="text/plain",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=requirements
)
