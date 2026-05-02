import docker
from actuator.scaler_interface import ScalerInterface

class DockerActuator(ScalerInterface):
    def __init__(self, config):
        # This connects to the Docker Desktop engine you have running
        self.client = docker.from_env()
        self.config = config
        self.image = self.config['scaling_rules']['target_service_image']
        self.prefix = "ai_sysadmin_worker_"
        
        # Pull the image so it doesn't lag on the first run
        print(f"Actuator: Ensuring image {self.image} is available...")
        try:
            self.client.images.pull(self.image)
        except Exception as e:
            print(f"⚠️ Actuator: Could not pull image '{self.image}': {e}. Will use cached version if available.")
    
    def get_workers(self):
        # Find all containers we are actively managing
        return self.client.containers.list(filters={"name": self.prefix})

    def scale_up(self):
        current_workers = len(self.get_workers())
        if current_workers < self.config['scaling_rules']['max_containers']:
            new_name = f"{self.prefix}{current_workers + 1}"
            print(f"⚙️ Actuator: Spinning up new container -> {new_name}")
            # detach=True means it runs in the background
            self.client.containers.run(self.image, name=new_name, detach=True)
            return True
        print("⚠️ Actuator: Max container limit reached. Cannot scale up.")
        return False

    def scale_down(self):
        workers = self.get_workers()
        if len(workers) > self.config['scaling_rules']['min_containers']:
            # Grab the last container spawned and kill it
            target = sorted(workers, key=lambda c: c.name)[-1]
            print(f"🛑 Actuator: Stopping and removing -> {target.name}")
            target.stop()
            target.remove()
            return True
        print("⚠️ Actuator: Minimum container limit reached. Cannot scale down.")
        return False