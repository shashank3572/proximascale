import docker

CONTAINER_NAME = "proximascale-app"

docker_client = docker.from_env()


def scale_up():
    print("SCALE UP requested")


def scale_down():
    print("SCALE DOWN requested")


if __name__ == "__main__":
    scale_up()
