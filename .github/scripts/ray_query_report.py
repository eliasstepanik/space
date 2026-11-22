"""Report whether the runner's software Vulkan device (lavapipe) exposes ray_query.

Reads vulkaninfo.json (from `vulkaninfo --json`) in the current directory and
prints one line per device. Report only: used by a `continue-on-error` CI step.
"""

import json

with open("vulkaninfo.json") as f:
    info = json.load(f)

found = False
for g in info.get("device_group", []):
    for dev in g.get("devices", []):
        name = dev.get("deviceName", "<unknown>")
        feats = dev.get("VkPhysicalDeviceFeatures2", {}).get("features", {})
        rq = feats.get("VkPhysicalDeviceRayQueryFeaturesKHR", {}).get("rayQuery")
        print(f"device: {name} (apiVersion={dev.get('apiVersion')}) ray_query={rq}")
        if "llvmpipe" in name.lower() or "lavapipe" in name.lower():
            found = True
            print(f"lavapipe ray_query: {rq}")

if not found:
    print("lavapipe: no llvmpipe/lavapipe device reported by vulkaninfo")
