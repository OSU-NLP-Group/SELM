def send_to_device(obj, device):
    if isinstance(obj, list):
        return [send_to_device(t, device) for t in obj]

    if isinstance(obj, tuple):
        return tuple(send_to_device(t, device) for t in obj)

    if isinstance(obj, dict):
        return {
            send_to_device(key, device): send_to_device(value, device)
            for key, value in obj.items()
        }

    if hasattr(obj, "to"):
        return obj.to(device)

    return obj
