def load_password(path="mqtt.pwd"):
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
            username, password = line.split()
            return username, password
    except Exception as e:
        print(f"Error loading password: {e}")
        return None, None
        