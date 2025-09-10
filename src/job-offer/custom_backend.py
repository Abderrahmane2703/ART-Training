"""
Custom SkyPilot backend with additional setup commands for dependencies
"""
import sky
from art.skypilot.backend import SkyPilotBackend
from art.skypilot.utils import to_thread_typed
import os


class CustomSkyPilotBackend(SkyPilotBackend):
    """Extended SkyPilotBackend with custom setup for our dependencies"""
    
    @classmethod
    async def initialize_cluster_with_setup(
        cls,
        *,
        cluster_name: str = "art",
        resources: sky.Resources | None = None,
        env_path: str | None = None,
        force_recreate: bool = False,
    ) -> "CustomSkyPilotBackend":
        """Initialize cluster with our custom dependencies"""
        
        # If we want to force recreate, first tear down existing cluster
        if force_recreate:
            try:
                print(f"Tearing down existing cluster {cluster_name}...")
                await to_thread_typed(
                    lambda: sky.down(cluster_names=[cluster_name], purge=True)
                )
                print(f"Cluster {cluster_name} torn down successfully")
            except Exception as e:
                print(f"No existing cluster to tear down or error: {e}")
        
        # Create a custom task with our setup script
        setup_script = """
set -euxo pipefail

# Ensure latest pip
python3 -m pip install --upgrade pip

# Install ART with backend extras (this brings vLLM/training stack with correct pins)
pip install "openpipe-art[backend]>=0.4.3"

# Minimal additional deps used by your rollout/judge and IO
pip install openai openpipe async-lru pydantic boto3 python-dotenv polars datasets
pip install uvicorn fastapi setproctitle tblib

echo "All dependencies installed successfully!"
"""
        
        # Ensure ports are properly configured
        if resources is None:
            resources = sky.Resources(
                accelerators="L4:1",
                disk_size=20,
                ports=["7999-8000"]  # Ensure ports are exposed
            )
        else:
            # Make sure ports 7999-8000 are included
            updated_ports = resources.ports if resources.ports else []
            if "7999-8000" not in updated_ports and "7999" not in updated_ports:
                updated_ports = ["7999-8000"] + list(updated_ports)
                resources = resources.copy(ports=updated_ports)
        
        # Create task with setup
        task = sky.Task(name=cluster_name)
        task.set_resources(resources)
        task.setup = setup_script
        
        # Load environment variables if provided
        envs = {}
        if env_path is not None:
            from dotenv import dotenv_values
            envs = dotenv_values(env_path)
            print(f"Loading envs from {env_path}")
            print(f"{len(envs)} environment variables found")
            task.update_envs(envs)
        
        # Launch the cluster with our custom setup
        print("Launching cluster with custom dependencies...")
        await to_thread_typed(
            lambda: sky.launch(task=task, cluster_name=cluster_name)
        )
        print("Cluster launched successfully with all dependencies!")
        
        # Now use the regular initialize_cluster to set up the ART server
        # This will detect the existing cluster and just start the ART server
        return await cls.initialize_cluster(
            cluster_name=cluster_name,
            resources=resources,
            env_path=env_path,
        )