//! Phase-0 spike 0.2: headless Vulkan capability probe.
//!
//! Creates a Vulkan instance WITHOUT any surface/window-system extensions so it
//! is safe to run on a machine with no display, enumerates physical devices and
//! prints a JSON report describing ray-tracing and video-encode capabilities.
//!
//! This is a throwaway probe used to decide CI test tiers and whether the H.264
//! encode spike (0.5) is worth running. It is intentionally standalone and does
//! not join the root `space` workspace.

use std::ffi::c_char;

use ash::vk;
use serde_json::{json, Value};

/// Convert a fixed-size NUL-terminated `[c_char; N]` (as used by Vulkan name
/// fields) into an owned `String`.
fn carray_to_string(arr: &[c_char]) -> String {
    let bytes: Vec<u8> = arr
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn device_type_str(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::OTHER => "other",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "unknown",
    }
}

fn probe_device(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Value {
    let props = unsafe { instance.get_physical_device_properties(pd) };

    let api_version = props.api_version;
    let api_version_str = format!(
        "{}.{}.{}",
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version),
        vk::api_version_patch(api_version)
    );

    // Enumerate device extension names once.
    let device_exts: Vec<String> = unsafe {
        instance
            .enumerate_device_extension_properties(pd)
            .unwrap_or_default()
            .into_iter()
            .map(|e| carray_to_string(&e.extension_name))
            .collect()
    };
    let has_ext = |name: &std::ffi::CStr| {
        let s = name.to_string_lossy();
        device_exts.iter().any(|n| n == &*s)
    };

    // Query the ray-query feature via a Features2 pNext chain.
    let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut features2 =
        vk::PhysicalDeviceFeatures2::default().push_next(&mut ray_query_features);
    unsafe { instance.get_physical_device_features2(pd, &mut features2) };
    let ray_query_supported = ray_query_features.ray_query != 0;

    // Memory heaps: total size across all reported heaps.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
    let heap_count = mem_props.memory_heap_count as usize;
    let heaps_total: u64 = mem_props.memory_heaps[..heap_count]
        .iter()
        .map(|h| h.size)
        .sum();
    let heaps: Vec<Value> = mem_props.memory_heaps[..heap_count]
        .iter()
        .map(|h| json!({ "size_bytes": h.size }))
        .collect();

    // Video encode: presence of the required extensions. ash 0.38.0 does not
    // expose get_physical_device_video_profile_info_khr, so we cannot enumerate
    // VK_VIDEO_H264_STANDARD_PROFILE_ID profiles here; we degrade gracefully and
    // report extension presence only.
    let video_encode_exts_present =
        has_ext(vk::KHR_VIDEO_ENCODE_QUEUE_NAME.as_ref()) && has_ext(vk::KHR_VIDEO_ENCODE_H264_NAME.as_ref());

    let video_encode = json!({
        "extensions_present": video_encode_exts_present,
        "encode_op_supported": null,
        "video_profiles": null,
        "limitation": "ash 0.38.0 does not expose get_physical_device_video_profile_info_khr; \
                       video profile (VK_VIDEO_H264_STANDARD_PROFILE_ID) encode-op support could \
                       not be queried. Only extension presence is reported.",
    });

    json!({
        "device_name": carray_to_string(&props.device_name),
        "vendor_id": props.vendor_id,
        "device_id": props.device_id,
        "device_type": device_type_str(props.device_type),
        "api_version": api_version_str,
        "api_version_raw": api_version,
        "driver_version": props.driver_version,
        "ray_query": {
            "feature_supported": ray_query_supported,
            "extensions": {
                "VK_KHR_ray_query": has_ext(vk::KHR_RAY_QUERY_NAME.as_ref()),
                "VK_KHR_acceleration_structure": has_ext(vk::KHR_ACCELERATION_STRUCTURE_NAME.as_ref()),
                "VK_KHR_deferred_host_operations": has_ext(vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.as_ref()),
                "VK_KHR_ray_tracing_pipeline": has_ext(vk::KHR_RAY_TRACING_PIPELINE_NAME.as_ref()),
            },
        },
        "rt_tier": {
            "ray_query_enableable": ray_query_supported,
            "note": "ray_query feature reported supported by the driver; equivalent to a minimal \
                     ray-query RT tier. Full RT pipeline also requires the ray-tracing extensions above.",
        },
        "video_encode_h264": video_encode,
        "vulkan_video_decode_h264": {
            "present": has_ext(vk::KHR_VIDEO_DECODE_H264_NAME.as_ref()),
            "note": "informational only",
        },
        "memory": {
            "heaps_total_bytes": heaps_total,
            "heaps": heaps,
        },
        "surface": {
            "requested": false,
            "note": "surface extensions were NOT requested; probe is headless and assumes no display.",
        },
        "all_device_extensions": device_exts,
    })
}

fn main() {
    // Load the Vulkan loader. On a machine with no Vulkan driver/loader this
    // returns Err; we report that as valid JSON rather than panicking.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(err) => {
            let out = json!({
                "vulkan_available": false,
                "loader_error": err.to_string(),
                "devices_found": 0,
                "devices": [],
                "surface": {
                    "requested": false,
                    "note": "surface extensions were NOT requested; probe is headless.",
                },
                "conclusion": "No Vulkan loader/driver is available in this environment. \
                               RT tests and the H.264 encode spike cannot run here.",
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
            // No driver is a valid, important result: exit 0 with valid JSON.
            return;
        }
    };

    // Create an instance with NO surface / window-system extensions (headless).
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vkprobe")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"vkprobe")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(err) => {
            let out = json!({
                "vulkan_available": true,
                "instance_created": false,
                "instance_error": format!("{err:?}"),
                "devices_found": 0,
                "devices": [],
                "surface": {
                    "requested": false,
                    "note": "surface extensions were NOT requested; probe is headless.",
                },
                "conclusion": "Vulkan loader present but instance creation failed \
                               (likely no usable ICD). RT tests and the H.264 encode \
                               spike cannot run here.",
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
            return;
        }
    };

    let physical_devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();

    let devices: Vec<Value> = physical_devices
        .iter()
        .map(|&pd| probe_device(&instance, pd))
        .collect();

    let out = json!({
        "vulkan_available": true,
        "instance_created": true,
        "ash_version": "0.38.0+1.3.281",
        "devices_found": devices.len(),
        "devices": devices,
        "surface": {
            "requested": false,
            "note": "surface extensions were NOT requested; probe is headless and assumes no display.",
        },
        "conclusion": if devices.is_empty() {
            "Vulkan loader and instance OK but zero physical devices: no GPU/ICD exposed. \
             RT tests and the H.264 encode spike cannot run here."
        } else {
            "Devices found; see per-device ray_query and video_encode_h264 fields for tier decision."
        },
    });

    println!("{}", serde_json::to_string_pretty(&out).unwrap());

    unsafe {
        instance.destroy_instance(None);
    }
}
