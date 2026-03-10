
/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_sdf.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyobj_loader_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <openxr/openxr.h>
#include <string>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

#include <cmrc/cmrc.hpp>

CMRC_DECLARE(ngp);

#ifdef copysign
#	undef copysign
#endif

#define M_PI 3.14159265359

namespace ngp {

static constexpr uint32_t MARCH_ITER = 10000;

Testbed::NetworkDims Testbed::network_dims_sphere_sdf() const {
	NetworkDims dims;
	dims.n_input = 7;
	dims.n_output = 1;
	dims.n_pos = 3;
	return dims;
}

__device__ inline float square(float x) { return x * x; }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline vec3 mix(const vec3& a, const vec3& b, float t) { return a + (b - a) * t; }

__device__ inline float SchlickFresnel(float u) {
	float m = __saturatef(1.0 - u);
	return square(square(m)) * m;
}

__device__ inline float G1(float NdotH, float a) {
	if (a >= 1.0) {
		return 1.0 / PI();
	}
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return (a2 - 1.0) / (PI() * log(a2) * t);
}

__device__ inline float G2(float NdotH, float a) {
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return a2 / (PI() * t * t);
}

__device__ inline float SmithG_GGX(float NdotV, float alphaG) {
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0 / (NdotV + sqrtf(a + b - a * b));
}

// this function largely based on:
// https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
// http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
__device__ vec3 sphere_evaluate_shading(
	const vec3& base_color,
	const vec3& ambient_color, // :)
	const vec3& light_color,   // :)
	float metallic,
	float subsurface,
	float specular,
	float roughness,
	float specular_tint,
	float sheen,
	float sheen_tint,
	float clearcoat,
	float clearcoat_gloss,
	vec3 L,
	vec3 V,
	vec3 N
) {
	float NdotL = dot(N, L);
	float NdotV = dot(N, V);

	vec3 H = normalize(L + V);
	float NdotH = dot(N, H);
	float LdotH = dot(L, H);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
	vec3 amb = (ambient_color * mix(0.2f, FV, metallic));
	amb *= base_color;
	if (NdotL < 0.f || NdotV < 0.f) {
		return amb;
	}

	float luminance = dot(base_color, vec3{0.3f, 0.6f, 0.1f});

	// normalize luminance to isolate hue and saturation components
	vec3 Ctint = base_color * (1.f / (luminance + 0.00001f));
	vec3 Cspec0 = mix(mix(vec3(1.0f), Ctint, specular_tint) * specular * 0.08f, base_color, metallic);
	vec3 Csheen = mix(vec3(1.0f), Ctint, sheen_tint);

	float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
	float Fd = mix(1, Fd90, FL) * mix(1.f, Fd90, FV);

	// Based on Hanrahan-Krueger BRDF approximation of isotropic BSSRDF
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LdotH * LdotH * roughness;
	float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - 0.5f) + 0.5f);

	// Specular
	float a = std::max(0.001f, square(roughness));
	float Ds = G2(NdotH, a);
	float FH = SchlickFresnel(LdotH);
	vec3 Fs = mix(Cspec0, vec3(1.0f), FH);
	float Gs = SmithG_GGX(NdotL, a) * SmithG_GGX(NdotV, a);

	// sheen
	vec3 Fsheen = FH * sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = G1(NdotH, mix(0.1f, 0.001f, clearcoat_gloss));
	float Fr = mix(0.04f, 1.0f, FH);
	float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

	float CCs = 0.25f * clearcoat * Gr * Fr * Dr;
	vec3 brdf = (float(1.0f / PI()) * mix(Fd, ss, subsurface) * base_color + Fsheen) * (1.0f - metallic) + Gs * Fs * Ds + vec3{CCs, CCs, CCs};
	return vec3(brdf * light_color) * NdotL + amb;
}

__global__ void sphere_advance_pos_kernel_sdf(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	BoundingBox aabb,
	float floor_y,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	int max_octree_depth,
	float distance_scale,
	float maximum_distance,
	float k,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	SdfPayload& payload = payloads[i];
	if (!payload.alive) {
		return;
	}

	float distance = distances[i] - zero_offset;

	distance *= distance_scale;

	// Advance by the predicted distance
	vec3 pos = positions[i];
	pos += distance * payload.dir;

	// Skip over regions not covered by the octree
	if (octree_nodes && !contains(octree_nodes, max_octree_depth, pos)) {
		float octree_distance = ray_intersect(octree_nodes, max_octree_depth, pos, payload.dir) + 1e-6f;
		distance += octree_distance;
		pos += octree_distance * payload.dir;
	}

	if (pos.y < floor_y && payload.dir.y < 0.f) {
		float floor_dist = -(pos.y - floor_y) / payload.dir.y;
		distance += floor_dist;
		pos += floor_dist * payload.dir;
		payload.alive = false;
	}

	positions[i] = pos;

	if (total_distances && distance > 0.0f) {
		// From https://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
		float total_distance = total_distances[i];
		float y = distance * distance / (2.0f * prev_distances[i]);
		float d = sqrtf(distance * distance - y * y);

		min_visibility[i] = fminf(min_visibility[i], k * d / fmaxf(0.0f, total_distance - y));
		prev_distances[i] = distance;
		total_distances[i] = total_distance + distance;
	}

	bool stay_alive = distance > maximum_distance && fabsf(distance / 2) > 3 * maximum_distance;
	if (!stay_alive) {
		payload.alive = false;
		return;
	}

	if (!aabb.contains(pos)) {
		payload.alive = false;
		return;
	}

	++payload.n_steps;
}

__global__ void sphere_perturb_sdf_samples(
	uint32_t n_elements, const vec3* __restrict__ perturbations, vec3* __restrict__ positions, float* __restrict__ distances
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	vec3 perturbation = perturbations[i];
	positions[i] += perturbation;

	// Small epsilon above 1 to ensure a triangle is always found.
	distances[i] = length(perturbation) * 1.001f;
}

__global__ void sphere_prepare_shadow_rays(
	const uint32_t n_elements,
	vec3 sun_dir,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility,
	SdfPayload* __restrict__ payloads,
	BoundingBox aabb,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	int max_octree_depth
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	SdfPayload& payload = payloads[i];

	// Step back a little along the ray to prevent self-intersection
	vec3 view_pos = positions[i] + normalize(faceforward(normals[i], payload.dir, normals[i])) * 1e-3f;
	vec3 dir = normalize(sun_dir);

	float t = fmaxf(aabb.ray_intersect(view_pos, dir).x + 1e-6f, 0.0f);
	view_pos += t * dir;

	if (octree_nodes && !contains(octree_nodes, max_octree_depth, view_pos)) {
		t = fmaxf(0.0f, ray_intersect(octree_nodes, max_octree_depth, view_pos, dir) + 1e-6f);
		view_pos += t * dir;
	}

	positions[i] = view_pos;

	if (!aabb.contains(view_pos)) {
		distances[i] = MAX_DEPTH();
		payload.alive = false;
		min_visibility[i] = 1.0f;
		return;
	}

	distances[i] = MAX_DEPTH();
	payload.idx = i;
	payload.dir = dir;
	payload.n_steps = 0;
	payload.alive = true;

	if (prev_distances) {
		prev_distances[i] = 1e20f;
	}

	if (total_distances) {
		total_distances[i] = 0.0f;
	}

	if (min_visibility) {
		min_visibility[i] = 1.0f;
	}
}

__global__ void sphere_write_shadow_ray_result(
	const uint32_t n_elements,
	BoundingBox aabb,
	const vec3* __restrict__ positions,
	const SdfPayload* __restrict__ shadow_payloads,
	const float* __restrict__ min_visibility,
	float* __restrict__ shadow_factors
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	shadow_factors[shadow_payloads[i].idx] = aabb.contains(positions[i]) ? 0.0f : min_visibility[i];
}

__global__ void sphere_shade_kernel_sdf(
	const uint32_t n_elements,
	BoundingBox aabb,
	float floor_y,
	const ERenderMode mode,
	const BRDFParams brdf,
	vec3 sun_dir,
	vec3 up_dir,
	mat4x3 camera_matrix,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	SdfPayload& payload = payloads[i];
	if (!aabb.contains(positions[i])) {
		return;
	}

	// The normal in memory isn't normalized yet
	vec3 normal = normalize(normals[i]);
	vec3 pos = positions[i];
	bool floor = false;
	if (pos.y < floor_y + 0.001f && payload.dir.y < 0.f) {
		normal = vec3{0.0f, 1.0f, 0.0f};
		floor = true;
	}

	vec3 cam_pos = camera_matrix[3];
	vec3 cam_fwd = camera_matrix[2];
	float ao = powf(0.92f, payload.n_steps * 0.5f) * (1.f / 0.92f);
	vec3 color;
	switch (mode) {
		case ERenderMode::AO: color = vec3(powf(0.92f, payload.n_steps)); break;
		case ERenderMode::Shade: {
			float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
			vec3 suncol = vec3{255.f / 255.0f, 225.f / 255.0f, 195.f / 255.0f} * 4.f *
				distances[i]; // Distance encodes shadow occlusion. 0=occluded, 1=no shadow
			const vec3 skycol = vec3{195.f / 255.0f, 215.f / 255.0f, 255.f / 255.0f} * 4.f * skyam;
			float check_size = 8.f / aabb.diag().x;
			float check = ((int(floorf(check_size * (pos.x - aabb.min.x))) ^ int(floorf(check_size * (pos.z - aabb.min.z)))) & 1) ? 0.8f :
																																	0.2f;
			const vec3 floorcol = vec3{check * check * check, check * check, check};
			color = sphere_evaluate_shading(
				floor ? floorcol : brdf.basecolor * brdf.basecolor,
				brdf.ambientcolor * skycol,
				suncol,
				floor ? 0.f : brdf.metallic,
				floor ? 0.f : brdf.subsurface,
				floor ? 1.f : brdf.specular,
				floor ? 0.5f : brdf.roughness,
				0.f,
				floor ? 0.f : brdf.sheen,
				0.f,
				floor ? 0.f : brdf.clearcoat,
				brdf.clearcoat_gloss,
				sun_dir,
				-normalize(payload.dir),
				normal
			);
		} break;
		case ERenderMode::Depth: color = vec3(dot(cam_fwd, pos - cam_pos)); break;
		case ERenderMode::Positions: {
			color = (pos - 0.5f) / 2.0f + 0.5f;
		} break;
		case ERenderMode::Normals: color = 0.5f * normal + 0.5f; break;
		case ERenderMode::Cost: color = vec3((float)payload.n_steps / 30); break;
		case ERenderMode::EncodingVis: color = normals[i]; break;
	}

	frame_buffer[payload.idx] = {color.r, color.g, color.b, 1.0f};
	depth_buffer[payload.idx] = dot(cam_fwd, pos - cam_pos);
}

__global__ void sphere_compact_kernel_shadow_sdf(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* src_positions,
	float* src_distances,
	SdfPayload* src_payloads,
	float* src_prev_distances,
	float* src_total_distances,
	float* src_min_visibility,
	vec3* dst_positions,
	float* dst_distances,
	SdfPayload* dst_payloads,
	float* dst_prev_distances,
	float* dst_total_distances,
	float* dst_min_visibility,
	vec3* dst_final_positions,
	float* dst_final_distances,
	SdfPayload* dst_final_payloads,
	float* dst_final_prev_distances,
	float* dst_final_total_distances,
	float* dst_final_min_visibility,
	BoundingBox aabb,
	uint32_t* counter,
	uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	SdfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
		dst_prev_distances[idx] = src_prev_distances[i];
		dst_total_distances[idx] = src_total_distances[i];
		dst_min_visibility[idx] = src_min_visibility[i];
	} else { // For shadow rays, collect _all_ final samples to keep track of their partial visibility
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = src_distances[i];
		dst_final_prev_distances[idx] = src_prev_distances[i];
		dst_final_total_distances[idx] = src_total_distances[i];
		dst_final_min_visibility[idx] = aabb.contains(src_positions[i]) ? 0.0f : src_min_visibility[i];
	}
}

__global__ void sphere_compact_kernel_sdf(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* src_positions,
	float* src_distances,
	SdfPayload* src_payloads,
	vec3* dst_positions,
	float* dst_distances,
	SdfPayload* dst_payloads,
	vec3* dst_final_positions,
	float* dst_final_distances,
	SdfPayload* dst_final_payloads,
	BoundingBox aabb,
	uint32_t* counter,
	uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	SdfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
	} else if (aabb.contains(src_positions[i])) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = 1.0f; // HACK: Distances encode shadowing factor when shading
	}
}

__global__ void sphere_uniform_octree_sample_kernel(
	const uint32_t num_elements,
	default_rng_t rng,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	uint32_t num_nodes,
	uint32_t depth,
	vec3* __restrict__ samples
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) {
		return;
	}

	rng.advance(i * (1 << 8));

	// Samples random nodes until a leaf is picked
	uint32_t node;
	uint32_t child;
	do {
		node = umin((uint32_t)(random_val(rng) * num_nodes), num_nodes - 1);
		child = umin((uint32_t)(random_val(rng) * 8), 8u - 1);
	} while (octree_nodes[node].depth < depth - 2 || octree_nodes[node].children[child] == -1);

	// Here it should be guaranteed that any child of the node is -1
	float size = scalbnf(1.0f, -depth + 1);

	u16vec3 pos = octree_nodes[node].pos * uint16_t(2);
	if (child & 1) {
		++pos.x;
	}
	if (child & 2) {
		++pos.y;
	}
	if (child & 4) {
		++pos.z;
	}
	samples[i] = size * (vec3(pos) + samples[i]);
}

__global__ void sphere_scale_to_aabb_kernel(uint32_t n_elements, BoundingBox aabb, vec3* __restrict__ inout) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	inout[i] = aabb.min + inout[i] * aabb.diag();
}

__global__ void sphere_compare_signs_kernel(
	uint32_t n_elements,
	const vec3* positions,
	const float* distances_ref,
	const float* distances_model,
	uint32_t* counters,
	const TriangleOctreeNode* octree_nodes,
	int max_octree_depth
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}
	bool inside1 = distances_ref[i] <= 0.f;
	bool inside2 = distances_model[i] <= 0.f;
	if (octree_nodes && !contains(octree_nodes, max_octree_depth, positions[i])) {
		inside2 = inside1;          // assume, when using the octree, that the model is always correct outside the octree.
		atomicAdd(&counters[6], 1); // outside the octree
	} else {
		atomicAdd(&counters[7], 1); // inside the octree
	}
	atomicAdd(&counters[inside1 ? 0 : 1], 1);
	atomicAdd(&counters[inside2 ? 2 : 3], 1);
	if (inside1 && inside2) {
		atomicAdd(&counters[4], 1);
	}
	if (inside1 || inside2) {
		atomicAdd(&counters[5], 1);
	}
}

__global__ void sphere_scale_iou_counters_kernel(uint32_t n_elements, uint32_t* counters, float scale) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	counters[i] = uint32_t(roundf(counters[i] * scale));
}

__global__ void sphere_assign_float(uint32_t n_elements, float value, float* __restrict__ out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	out[i] = value;
}

__global__ void sphere_init_rays_with_payload_kernel_sdf(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	SdfPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Lens lens,
	const TriangleOctreeNode* __restrict__ octree_nodes = nullptr,
	int max_octree_depth = 0
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens
	);

	distances[idx] = MAX_DEPTH();
	depth_buffer[idx] = MAX_DEPTH();

	SdfPayload& payload = payloads[idx];

	if (!ray.is_valid()) {
		payload.dir = ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.dir = (1.0f / n) * ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o - plane_z * ray.d;
		depth_buffer[idx] = -plane_z;
		return;
	}

	ray.d = normalize(ray.d);
	float t = max(aabb.ray_intersect(ray.o, ray.d).x, 0.0f);

	ray.advance(t + 1e-6f);

	if (octree_nodes && !contains(octree_nodes, max_octree_depth, ray.o)) {
		t = max(0.0f, ray_intersect(octree_nodes, max_octree_depth, ray.o, ray.d));
		if (ray.o.y > floor_y && ray.d.y < 0.f) {
			float floor_dist = -(ray.o.y - floor_y) / ray.d.y;
			if (floor_dist > 0.f) {
				t = min(t, floor_dist);
			}
		}

		ray.advance(t + 1e-6f);
	}

	positions[idx] = ray.o;

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

	payload.dir = ray.d;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = aabb.contains(ray.o);
}

__host__ __device__ uint32_t sphere_sample_discrete(float uniform_sample, const float* __restrict__ cdf, int length) {
	return binary_search(uniform_sample, cdf, length);
}

__global__ void sphere_sample_uniform_on_triangle_kernel(
	uint32_t n_elements, const float* __restrict__ cdf, uint32_t length, const Triangle* __restrict__ triangles, vec3* __restrict__ sampled_positions
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	vec3 sample = sampled_positions[i];
	uint32_t tri_idx = sphere_sample_discrete(sample.x, cdf, length);

	sampled_positions[i] = triangles[tri_idx].sample_uniform_position(sample.yz());
}

void Testbed::load_sphere(const fs::path& data_path) {
	tlog::info() << "Trying to load sphere at: " << data_path;
	std::ifstream file(data_path.str());
	if (!file.is_open()) {
		tlog::error() << "Could not open sphere .bezdat file: " << data_path;
		return;
	}

	std::string line;

	if (!std::getline(file, line)) {
		tlog::error() << "Sphere .bezdat file is empty.";
		return;
	}

	std::stringstream ss(line);

	m_raw_aabb.min = vec3(std::numeric_limits<float>::infinity());
	m_raw_aabb.max = vec3(-std::numeric_limits<float>::infinity());

	m_raw_aabb.enlarge(vec3(0.0, 0.0, 0.0));
	m_raw_aabb.enlarge(vec3(1.0, 1.0, 1.0));

	const float inflation = 0.05f;

	m_raw_aabb.inflate(length(m_raw_aabb.diag()) * inflation);
	float scale = max(m_raw_aabb.diag());

	m_aabb = {};
	m_aabb.enlarge(vec3(0.0, 0.0, 0.0));
	m_aabb.enlarge(vec3(1.0, 1.0, 1.0));
	
	m_aabb.inflate(length(m_aabb.diag()) * inflation);

	m_render_aabb = m_aabb;
	m_render_aabb.inflate(length(m_render_aabb.diag()) * inflation);
	m_render_aabb_to_local = mat3(1.0f);
}

__host__ __device__
vec3 point_on_sphere(const vec3& center, float radius, float z, float phi) {
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z*z));
    vec3 dir = { r_xy * cosf(phi), r_xy * sinf(phi), z };
    return center + radius * dir;
}

inline __device__ vec3 random_point_in_cube(
    default_rng_t& rng,
    float min_val,
    float max_val
) {
    const float extent = max_val - min_val;

    return {
        min_val + random_val(rng) * extent,
        min_val + random_val(rng) * extent,
        min_val + random_val(rng) * extent
    };
}

__device__ inline float sample_radius_volume_uniform(default_rng_t& rng, float rmin, float rmax) {
    float t = random_val(rng);
    float rmin3 = rmin*rmin*rmin;
    float rmax3 = rmax*rmax*rmax;
    return cbrtf(rmin3 + t * (rmax3 - rmin3));
}

__device__ inline vec3 random_unit_vector(default_rng_t& rng) {
    // uniform on sphere
    const float u = random_val(rng);          // [0,1)
    const float v = random_val(rng);          // [0,1)
    const float z = 2.0f * u - 1.0f;          // [-1,1]
    const float a = 2.0f * (float)M_PI * v;   // [0,2pi)
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
    return { r * cosf(a), r * sinf(a), z };
}

__device__ inline float clamp_sd(float sd) {
    const float dmax = 0.25f; 
    return copysignf(fminf(fabsf(sd), dmax), sd);
}

__device__ inline vec3 nearest_point_on_sphere(const vec3& p, const vec3& c, float R) {
    vec3 d = p - c;
    float len2 = d.x*d.x + d.y*d.y + d.z*d.z;

    if (len2 <= 1e-20f) {
        return { c.x + R, c.y, c.z };
    }

    float inv_len = rsqrtf(len2);
    return c + d * (R * inv_len);
}

__global__ void generate_sphere_surface_distances(
    uint32_t n_elements,
    ngp::Testbed::Sphere::TrainingInput* __restrict__ inputs,
    float* __restrict__ distances,
    uint32_t seed,
	uint32_t rng_offset,
    float aabb_min,
    float aabb_max
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    default_rng_t rng{seed};
    rng.advance((i) * 7);

    const float min_R = 0.01f;
    const float max_R = 0.25f;

    float x = random_val(rng);
    float y = random_val(rng);
    float z = random_val(rng);
    const vec3 center = {x, y, z};

    float R = min_R + random_val(rng) * (max_R - min_R);
	
    const vec3 random_sample = random_point_in_cube(rng, aabb_min, aabb_max);
	const vec3 sample = nearest_point_on_sphere(random_sample, center, R);

    float sd = length(sample - center) - R;

    auto& out = inputs[i];
    out.x = sample.x; out.y = sample.y; out.z = sample.z;
    out.r = R;
    out.pos_x = center.x; out.pos_y = center.y; out.pos_z = center.z;

    distances[i] = sd;	

	// printf("Center: (%f, %f, %f), Radius: %f, Sample: (%f, %f, %f) SD: %f\n",
	// 	center.x, center.y, center.z, R,
	// 	sample.x, sample.y, sample.z, sd
	// );
}

__global__ void generate_sphere_uniform_samples(
    uint32_t n_elements,
    ngp::Testbed::Sphere::TrainingInput* __restrict__ inputs,
    float* __restrict__ distances,
    uint32_t seed,
	uint32_t rng_offset,
    float aabb_min,
    float aabb_max
) {
 	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    default_rng_t rng{seed};
    rng.advance((i) * 7);

    const float min_R = 0.01f;
    const float max_R = 0.25f;

	float x = random_val(rng);
	x = ((int) roundf((x * 1000))) / 1000.0f;
    float y = random_val(rng);
	y = ((int) roundf((y * 1000))) / 1000.0f;
    float z = random_val(rng);
	z = ((int) roundf((z * 1000))) / 1000.0f;
    const vec3 center = {x, 0.5, 0.5};

    float R = min_R + random_val(rng) * (max_R - min_R);
	
    const vec3 sample = random_point_in_cube(rng, aabb_min, aabb_max);

    float sd = length(sample - center) - R;

    auto& out = inputs[i];
    out.x = sample.x; out.y = sample.y; out.z = sample.z;
    out.r = R;
    out.pos_x = center.x; out.pos_y = center.y; out.pos_z = center.z;

    distances[i] = sd;

	// printf("Center: (%f, %f, %f), Radius: %f, Sample: (%f, %f, %f) SD: %f\n",
	// 	center.x, center.y, center.z, R,
	// 	sample.x, sample.y, sample.z, sd
	// );

}

void Testbed::generate_training_samples_sphere(Sphere::TrainingInput* inputs, float* distances, uint32_t n_to_generate, cudaStream_t stream, bool uniform_only) {
	uint32_t n_to_generate_base = n_to_generate / 16;
	const uint32_t n_to_generate_uniform = uniform_only ? n_to_generate : n_to_generate_base * 8;
	const uint32_t n_to_generate_uniform_surface = uniform_only ? n_to_generate : n_to_generate_base * 8;

	uint32_t off_uniform = 0;
	uint32_t off_surface = n_to_generate_uniform;

	linear_kernel(
		generate_sphere_uniform_samples,
		0,
		stream,
		n_to_generate_uniform,
		inputs,
		distances,
		m_training_step + 1,
		off_uniform,
		-0.00,
		1.00
	);


	linear_kernel(
		generate_sphere_surface_distances,
		0,
		stream,
		n_to_generate_uniform_surface,
		inputs + n_to_generate_uniform,
		distances + n_to_generate_uniform,
		m_training_step + 1,      // seed
		off_surface,
		-0.00,
		1.00
	);

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void Testbed::train_sphere(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 1;
	const uint32_t n_input_dims = 7;

	if (m_sphere_sdf.training.size >= target_batch_size) {
		// Auxiliary matrices for training
		const uint32_t batch_size = (uint32_t)std::min(m_sphere_sdf.training.size, target_batch_size);

		// Permute all training records to de-correlate training data
		linear_kernel(
			shuffle<Sphere::TrainingInput>,
			0,
			stream,
			m_sphere_sdf.training.size,
			1,
			m_training_step,
			m_sphere_sdf.training.inputs.data(),
			m_sphere_sdf.training.inputs_shuffled.data()
		);
		linear_kernel(
			shuffle<float>,
			0,
			stream,
			m_sphere_sdf.training.size,
			1,
			m_training_step,
			m_sphere_sdf.training.distances.data(),
			m_sphere_sdf.training.distances_shuffled.data()
		);

		GPUMatrix<float> training_target_matrix(m_sphere_sdf.training.distances_shuffled.data(), n_output_dims, batch_size);
		GPUMatrix<float> training_batch_matrix((float*)(m_sphere_sdf.training.inputs_shuffled.data()), n_input_dims, batch_size);

		auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

		m_training_step++;

		if (get_loss_scalar) {
			m_loss_scalar.update(m_trainer->loss(stream, *ctx));
		}
	}
}

void Testbed::training_prep_sphere(uint32_t batch_size, cudaStream_t stream) {
	if (m_sphere_sdf.training.generate_sdf_data_online) {
		m_sphere_sdf.training.size = batch_size;
		m_sphere_sdf.training.inputs.enlarge(m_sphere_sdf.training.size);
		m_sphere_sdf.training.inputs_shuffled.enlarge(m_sphere_sdf.training.size);
		m_sphere_sdf.training.distances.enlarge(m_sphere_sdf.training.size);
		m_sphere_sdf.training.distances_shuffled.enlarge(m_sphere_sdf.training.size);

		generate_training_samples_sphere(
			m_sphere_sdf.training.inputs.data(), m_sphere_sdf.training.distances.data(), batch_size, stream, true
		);
	}
}


void Testbed::SphereTracerSphere::init_rays_from_camera(
	uint32_t sample_index,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Buffer2DView<const vec4>& envmap,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	const Lens& lens,
	const TriangleOctree* octree,
	uint32_t n_octree_levels,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x * resolution.y;
	enlarge(n_pixels, stream);

	const dim3 threads = {16, 8, 1};
	const dim3 blocks = {div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1};
	sphere_init_rays_with_payload_kernel_sdf<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].pos,
		m_rays[0].distance,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		aabb,
		floor_y,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask,
		lens,
		octree ? octree->nodes_gpu() : nullptr,
		octree ? n_octree_levels : 0
	);
	m_n_rays_initialized = (uint32_t)n_pixels;
	m_resolution = resolution;
}

void Testbed::SphereTracerSphere::init_rays_from_data(uint32_t n_elements, const RaysSdfSoa& data, cudaStream_t stream) {
	enlarge(n_elements, stream);
	m_rays[0].copy_from_other_async(n_elements, data, stream);
	m_n_rays_initialized = n_elements;
	m_resolution = {(int)m_n_rays_initialized, 1};
}

uint32_t Testbed::SphereTracerSphere::trace_bvh(TriangleBvh* bvh, const Triangle* triangles, cudaStream_t stream) {
	uint32_t n_alive = m_n_rays_initialized;
	m_n_rays_initialized = 0;

	if (!bvh) {
		return 0;
	}

	// Abuse the normal buffer to temporarily hold ray directions
	parallel_for_gpu(stream, n_alive, [payloads = m_rays[0].payload, normals = m_rays[0].normal] __device__(size_t i) {
		normals[i] = payloads[i].dir;
	});

	bvh->ray_trace_gpu(n_alive, m_rays[0].pos, m_rays[0].normal, triangles, stream);
	return n_alive;
}

void Testbed::upload_segment_extra_dims_sphere(
	cudaStream_t stream
) {
	float h[4];
    const auto& radius = m_sphere_sdf.m_current_render_target_radius;
    const auto& pos = m_sphere_sdf.m_current_render_target_pos;

    int i = 0;
    h[i++] = radius;
	h[i++] = pos.x;
	h[i++] = pos.y;
	h[i++] = pos.z;

    if (m_sphere_sdf.extra_dims.size() != 4) {
        m_sphere_sdf.extra_dims.resize(4);
    }
	
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		m_sphere_sdf.extra_dims.data(),
		h,
		4 * sizeof(float),
		cudaMemcpyHostToDevice,
		stream
	));
}

uint32_t Testbed::SphereTracerSphere::trace(
	const distance_fun_spline_t& distance_function,
	const Network<float, network_precision_t>* network,
	float zero_offset,
	float distance_scale,
	float maximum_distance,
	const BoundingBox& aabb,
	const float floor_y,
	const TriangleOctree* octree,
	const uint32_t n_octree_levels,
	cudaStream_t stream
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}

	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

	const uint32_t STEPS_INBETWEEN_COMPACTION = 4;

	uint32_t n_alive = m_n_rays_initialized;
	m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		// Compact more frequently in the first couple of steps
		uint32_t step_size = std::min(i, STEPS_INBETWEEN_COMPACTION);

		RaysSdfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		RaysSdfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;

		if (m_fused_trace_kernel) {
			dim3 threads, blocks;
			if (m_resolution.x >= 8 && m_resolution.y >= 16) {
				threads = {8, 16, 1};
				blocks = {
					div_round_up((uint32_t)m_resolution.x, threads.x), div_round_up((uint32_t)m_resolution.y, threads.y), 1
				};
			} else {
				threads = {N_THREADS_LINEAR, 1, 1};
				blocks = {n_blocks_linear(n_alive, threads.x), 1, 1};
			}
			m_owner->upload_segment_extra_dims_sphere(stream);
			float* extra_dim_ptr = m_owner->m_sphere_sdf.extra_dims.data();
			
			m_fused_trace_kernel->launch(
				blocks,
				threads,
				0,
				stream,
				m_resolution,
				zero_offset,
				rays_tmp.pos,
				rays_tmp.distance,
				rays_tmp.payload,
				aabb,
				floor_y,
				octree ? octree->nodes_gpu() : nullptr,
				octree ? n_octree_levels : 0,
				distance_scale,
				maximum_distance,
				m_shadow_sharpness,
				m_trace_shadow_rays ? rays_tmp.prev_distance : nullptr,
				m_trace_shadow_rays ? rays_tmp.total_distance : nullptr,
				m_trace_shadow_rays ? rays_tmp.min_visibility : nullptr,
				extra_dim_ptr,
				network->inference_params()
			);
		}

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
			if (m_trace_shadow_rays) {
				linear_kernel(
					sphere_compact_kernel_shadow_sdf,
					0,
					stream,
					n_alive,
					zero_offset,
					rays_tmp.pos,
					rays_tmp.distance,
					rays_tmp.payload,
					rays_tmp.prev_distance,
					rays_tmp.total_distance,
					rays_tmp.min_visibility,
					rays_current.pos,
					rays_current.distance,
					rays_current.payload,
					rays_current.prev_distance,
					rays_current.total_distance,
					rays_current.min_visibility,
					m_rays_hit.pos,
					m_rays_hit.distance,
					m_rays_hit.payload,
					m_rays_hit.prev_distance,
					m_rays_hit.total_distance,
					m_rays_hit.min_visibility,
					aabb,
					m_alive_counter,
					m_hit_counter
				);
			} else {
				linear_kernel(
					sphere_compact_kernel_sdf,
					0,
					stream,
					n_alive,
					zero_offset,
					rays_tmp.pos,
					rays_tmp.distance,
					rays_tmp.payload,
					rays_current.pos,
					rays_current.distance,
					rays_current.payload,
					m_rays_hit.pos,
					m_rays_hit.distance,
					m_rays_hit.payload,
					aabb,
					m_alive_counter,
					m_hit_counter
				);
			}
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		for (uint32_t j = 0; j < step_size; ++j) {
			distance_function(n_alive, rays_current.pos, rays_current.distance, stream);
			linear_kernel(
				sphere_advance_pos_kernel_sdf,
				0,
				stream,
				n_alive,
				zero_offset,
				rays_current.pos,
				rays_current.distance,
				rays_current.payload,
				aabb,
				floor_y,
				octree ? octree->nodes_gpu() : nullptr,
				octree ? n_octree_levels : 0,
				distance_scale,
				maximum_distance,
				m_shadow_sharpness,
				m_trace_shadow_rays ? rays_current.prev_distance : nullptr,
				m_trace_shadow_rays ? rays_current.total_distance : nullptr,
				m_trace_shadow_rays ? rays_current.min_visibility : nullptr
			);
		}

		i += step_size;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::SphereTracerSphere::enlarge(size_t n_elements, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3,
		vec3,
		float,
		float,
		float,
		float,
		SdfPayload, // m_rays[0]
		vec3,
		vec3,
		float,
		float,
		float,
		float,
		SdfPayload, // m_rays[1]
		vec3,
		vec3,
		float,
		float,
		float,
		float,
		SdfPayload, // m_rays_hit

		uint32_t,
		uint32_t>(
		stream,
		&m_scratch_alloc,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(
		std::get<0>(scratch),
		std::get<1>(scratch),
		std::get<2>(scratch),
		std::get<3>(scratch),
		std::get<4>(scratch),
		std::get<5>(scratch),
		std::get<6>(scratch)
	);
	m_rays[1].set(
		std::get<7>(scratch),
		std::get<8>(scratch),
		std::get<9>(scratch),
		std::get<10>(scratch),
		std::get<11>(scratch),
		std::get<12>(scratch),
		std::get<13>(scratch)
	);
	m_rays_hit.set(
		std::get<14>(scratch),
		std::get<15>(scratch),
		std::get<16>(scratch),
		std::get<17>(scratch),
		std::get<18>(scratch),
		std::get<19>(scratch),
		std::get<20>(scratch)
	);

	m_hit_counter = std::get<21>(scratch);
	m_alive_counter = std::get<22>(scratch);
}



void Testbed::render_sphere(
	cudaStream_t stream,
	CudaDevice& device,
	const distance_fun_spline_t& distance_function,
	const normals_fun_spline_t& normals_function,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	const Lens& lens,
	int visualized_dimension
) {
	auto jit_guard = m_network->jit_guard(stream, true);

	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}
	auto* octree_ptr = m_sphere_sdf.uses_takikawa_encoding || m_sphere_sdf.use_triangle_octree ? m_sphere_sdf.triangle_octree.get() : nullptr;

	SphereTracerSphere tracer{this};

	uint32_t n_octree_levels = octree_ptr ? octree_ptr->depth() : 0;

	BoundingBox sdf_bounding_box = m_aabb;
	sdf_bounding_box.inflate(m_sphere_sdf.zero_offset);

	if (m_jit_fusion) {
		if (!device.fused_render_kernel()) {
			try {
				device.set_fused_render_kernel(
					std::make_unique<CudaRtcKernel>(
						"trace_sphere",
						fmt::format(
							"{}\n#include <neural-graphics-primitives/fused_kernels/trace_sphere.cuh>\n",
							m_network->generate_device_function("eval_sdf")
						),
						all_files(cmrc::ngp::get_filesystem())
					)
				);
			} catch (const std::runtime_error& e) {
				tlog::warning() << e.what();
				tlog::warning() << "Disabling JIT fusion.";
				m_jit_fusion = false;
			}
		}

		if (device.fused_render_kernel()) {
			tracer.set_fused_trace_kernel(device.fused_render_kernel());
		}
	}

	tracer.init_rays_from_camera(
		render_buffer.spp,
		render_buffer.resolution,
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		sdf_bounding_box,
		get_floor_y(),
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		m_envmap.inference_view(),
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		lens,
		octree_ptr,
		n_octree_levels,
		stream
	);

	bool gt_raytrace = m_render_ground_truth && m_sphere_sdf.groundtruth_mode == ESDFGroundTruthMode::RaytracedMesh;

	auto trace = [&](SphereTracerSphere& tracer) {
		return tracer.trace(
			distance_function,
			m_network.get(),
			m_sphere_sdf.zero_offset,
			m_sphere_sdf.distance_scale,
			m_sphere_sdf.maximum_distance,
			sdf_bounding_box,
			get_floor_y(),
			octree_ptr,
			n_octree_levels,
			stream
		);
	};

	uint32_t n_hit;
	if (m_render_mode == ERenderMode::Slice) {
		n_hit = tracer.n_rays_initialized();
	} else {
		n_hit = trace(tracer);
	}
	RaysSdfSoa& rays_hit = m_render_mode == ERenderMode::Slice || gt_raytrace ? tracer.rays_init() : tracer.rays_hit();
	if (m_render_mode == ERenderMode::Slice) {
		if (visualized_dimension == -1) {
			distance_function(n_hit, rays_hit.pos, rays_hit.distance, stream);
			extract_dimension_pos_neg_kernel<float><<<n_blocks_linear(n_hit * 3), N_THREADS_LINEAR, 0, stream>>>(
				n_hit * 3, 0, 1, 3, rays_hit.distance, CM, (float*)rays_hit.normal
			);
		} else {
			// Store colors in the normal buffer
			uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);

			GPUMatrix<float> positions_matrix((float*)rays_hit.pos, 3, n_elements);
			GPUMatrix<float> colors_matrix((float*)rays_hit.normal, 3, n_elements);
			m_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, colors_matrix);
		}
	}
	ERenderMode render_mode = (visualized_dimension > -1 || m_render_mode == ERenderMode::Slice) ? ERenderMode::EncodingVis : m_render_mode;
	if (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Normals) {
		if (m_sphere_sdf.analytic_normals || gt_raytrace) {
			normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);
		} else {
			float fd_normals_epsilon = m_sphere_sdf.fd_normals_epsilon;

			FiniteDifferenceNormalsApproximator fd_normals;
			fd_normals.normal(n_hit, distance_function, rays_hit.pos, rays_hit.normal, fd_normals_epsilon, stream);
		}

		if (render_mode == ERenderMode::Shade && n_hit > 0) {
			// Shadow rays towards the sun
			SphereTracerSphere shadow_tracer{this};

			shadow_tracer.init_rays_from_data(n_hit, rays_hit, stream);
			shadow_tracer.set_fused_trace_kernel(tracer.fused_trace_kernel());
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_sphere_sdf.shadow_sharpness);
			RaysSdfSoa& shadow_rays_init = shadow_tracer.rays_init();
			linear_kernel(
				sphere_prepare_shadow_rays,
				0,
				stream,
				n_hit,
				normalize(m_sun_dir),
				shadow_rays_init.pos,
				shadow_rays_init.normal,
				shadow_rays_init.distance,
				shadow_rays_init.prev_distance,
				shadow_rays_init.total_distance,
				shadow_rays_init.min_visibility,
				shadow_rays_init.payload,
				sdf_bounding_box,
				octree_ptr ? octree_ptr->nodes_gpu() : nullptr,
				n_octree_levels
			);

			uint32_t n_hit_shadow = trace(shadow_tracer);
			auto& shadow_rays_hit = gt_raytrace ? shadow_tracer.rays_init() : shadow_tracer.rays_hit();

			linear_kernel(
				sphere_write_shadow_ray_result,
				0,
				stream,
				n_hit_shadow,
				sdf_bounding_box,
				shadow_rays_hit.pos,
				shadow_rays_hit.payload,
				shadow_rays_hit.min_visibility,
				rays_hit.distance
			);

			// todo: Reflection rays?
		}
	} else if (render_mode == ERenderMode::EncodingVis && m_render_mode != ERenderMode::Slice) {
		// HACK: Store colors temporarily in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);

		GPUMatrix<float> positions_matrix((float*)rays_hit.pos, 3, n_elements);
		GPUMatrix<float> colors_matrix((float*)rays_hit.normal, 3, n_elements);
		m_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, colors_matrix);
	}

	linear_kernel(
		sphere_shade_kernel_sdf,
		0,
		stream,
		n_hit,
		m_aabb,
		get_floor_y(),
		render_mode,
		m_sphere_sdf.brdf,
		normalize(m_sun_dir),
		normalize(m_up_dir),
		camera_matrix,
		rays_hit.pos,
		rays_hit.normal,
		rays_hit.distance,
		rays_hit.payload,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer
	);

	if (render_mode == ERenderMode::Cost) {
		std::vector<SdfPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(
			cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(SdfPayload), cudaMemcpyDeviceToHost, stream)
		);
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}

		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps / (float)n_hit);
	}
}



} // namespace ngp
