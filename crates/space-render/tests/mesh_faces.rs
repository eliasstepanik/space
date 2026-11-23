//! The generated meshes are solids whose faces look outwards.
//!
//! The renderer draws every triangle from both sides and decides by the
//! vertex normal which side a fragment shows (`side_of` in
//! `shaders/forward.wgsl`): the side the normal points to is the outside, the
//! other one the inside of the solid — unlit by the environment. A face whose
//! normal pointed inwards would therefore be dark from outside, and shadow
//! rays, which go by the winding, would let light through it. So: for every
//! triangle of every generator, the winding and the normals agree, and both
//! point away from the solid.
//!
//! No GPU needed.

use space_render::{sample_character, sample_ship, MeshData, MeshMaterial};

fn material() -> MeshMaterial {
    MeshMaterial {
        base_color: [1.0; 3],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        base_color_texture: None,
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f32; 3]) -> [f32; 3] {
    let length = dot(a, a).sqrt();
    assert!(length > 0.0, "a zero vector has no direction");
    a.map(|c| c / length)
}

/// One triangle: its corners, its vertex normals, and the unit normal its
/// winding gives it (counter-clockwise seen from the side that points to).
struct Triangle {
    corners: [[f32; 3]; 3],
    normals: [[f32; 3]; 3],
    winding: [f32; 3],
}

fn triangles(mesh: &MeshData) -> Vec<Triangle> {
    let (indices, rest) = mesh.indices.as_chunks::<3>();
    assert!(rest.is_empty(), "the index count is a multiple of three");
    indices
        .iter()
        .map(|t| {
            let vertex = |i: u32| &mesh.vertices[i as usize];
            let corners = t.map(|i| vertex(i).position);
            let normals = t.map(|i| vertex(i).normal);
            let winding = unit(cross(
                sub(corners[1], corners[0]),
                sub(corners[2], corners[0]),
            ));
            Triangle {
                corners,
                normals,
                winding,
            }
        })
        .collect()
}

/// Every triangle of `mesh` has unit vertex normals on the side its winding
/// faces.
fn assert_normals_follow_the_winding(name: &str, mesh: &MeshData) {
    for (i, triangle) in triangles(mesh).iter().enumerate() {
        for normal in triangle.normals {
            assert!(
                (dot(normal, normal) - 1.0).abs() < 1e-3,
                "{name}, triangle {i}: normal {normal:?} is not a unit vector"
            );
            assert!(
                dot(normal, triangle.winding) > 0.0,
                "{name}, triangle {i}: normal {normal:?} points to the other side than its \
                 winding ({:?})",
                triangle.winding
            );
        }
    }
}

#[test]
fn every_face_of_a_cuboid_points_away_from_its_centre() {
    // A cube, a floor slab, a wall panel, a landing pad.
    for half in [
        [1.0f32, 1.0, 1.0],
        [19.0, 0.1, 19.0],
        [0.08, 1.6, 1.6],
        [3.2, 0.12, 3.2],
    ] {
        let name = format!("cuboid {half:?}");
        let mesh = MeshData::cuboid(half, 1.0, material());
        assert_eq!(mesh.triangle_count(), 12, "{name}");
        // Two triangles for each of the six directions.
        let mut faces = [0usize; 6];
        for (i, triangle) in triangles(&mesh).iter().enumerate() {
            let w = triangle.winding;
            let axis = (0..3)
                .find(|&a| (w[a].abs() - 1.0).abs() < 1e-5)
                .unwrap_or_else(|| panic!("{name}, triangle {i}: {w:?} is not along an axis"));
            faces[2 * axis + usize::from(w[axis] < 0.0)] += 1;
            for (corner, normal) in triangle.corners.iter().zip(triangle.normals) {
                // The solid's centre is the origin: a corner of the +x face
                // is at x = +half, and so on.
                assert!(
                    (dot(*corner, w) - half[axis]).abs() < 1e-5,
                    "{name}, triangle {i}: corner {corner:?} is not on the face its winding \
                     ({w:?}) looks out of"
                );
                assert!(
                    sub(normal, w).iter().all(|c| c.abs() < 1e-5),
                    "{name}, triangle {i}: normal {normal:?}, winding {w:?}"
                );
            }
        }
        assert_eq!(faces, [2; 6], "{name}");
        assert!(mesh.is_closed(), "{name}");
        let volume = 8.0 * f64::from(half[0]) * f64::from(half[1]) * f64::from(half[2]);
        assert!(
            (mesh.signed_volume() - volume).abs() < 1e-3 * volume,
            "{name}: volume {} instead of {volume}",
            mesh.signed_volume()
        );
    }
}

#[test]
fn the_plane_faces_up_and_is_open() {
    let mesh = MeshData::plane([3.0, 2.0], [1.0, 1.0], material());
    assert_eq!(mesh.triangle_count(), 2);
    for triangle in triangles(&mesh) {
        assert!(sub(triangle.winding, [0.0, 1.0, 0.0])
            .iter()
            .all(|c| c.abs() < 1e-6));
    }
    assert_normals_follow_the_winding("plane", &mesh);
    // A sheet: both its sides face the open, and the renderer shades both.
    assert!(!mesh.is_closed());
}

#[test]
fn the_sample_models_are_closed_and_wound_and_shaded_outwards() {
    for (name, mesh) in [("ship", sample_ship()), ("character", sample_character())] {
        // Closed pieces with a positive volume are wound counter-clockwise
        // seen from outside; the normals then say the same.
        assert!(mesh.is_closed(), "{name}");
        assert!(mesh.signed_volume() > 0.0, "{name}");
        assert_normals_follow_the_winding(name, &mesh);
    }
}
