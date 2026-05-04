import docker

def cleanup_environment():
    client = docker.from_env()
    prefix = "proximascale_worker_"
    
    # Find all containers (even stopped ones) with our project prefix
    containers = client.containers.list(all=True, filters={"name": prefix})
    
    if not containers:
        print("✨ No project containers found. Everything is already clean!")
        return

    print(f"🧹 Found {len(containers)} containers. Starting cleanup...")

    for container in containers:
        try:
            print(f"Removing {container.name}...")
            # 'force=True' stops and removes the container in one go
            container.remove(force=True)
        except Exception as e:
            print(f"❌ Failed to remove {container.name}: {e}")

    print("✅ Cleanup complete. Docker is fresh.")

if __name__ == "__main__":
    cleanup_environment()