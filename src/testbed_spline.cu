
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

namespace ngp {

static constexpr uint32_t MARCH_ITER = 10000;

Testbed::NetworkDims Testbed::network_dims_spline_sdf() const {
	NetworkDims dims;
	dims.n_input = 3;
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
__device__ vec3 spline_evaluate_shading(
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

__global__ void spline_advance_pos_kernel_sdf(
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

__global__ void spline_perturb_sdf_samples(
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

__global__ void spline_prepare_shadow_rays(
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

__global__ void spline_write_shadow_ray_result(
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

__global__ void spline_shade_kernel_sdf(
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
			color = spline_evaluate_shading(
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

__global__ void spline_compact_kernel_shadow_sdf(
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

__global__ void spline_compact_kernel_sdf(
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

__global__ void spline_uniform_octree_sample_kernel(
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

__global__ void spline_scale_to_aabb_kernel(uint32_t n_elements, BoundingBox aabb, vec3* __restrict__ inout) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	inout[i] = aabb.min + inout[i] * aabb.diag();
}

__global__ void spline_compare_signs_kernel(
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

__global__ void spline_scale_iou_counters_kernel(uint32_t n_elements, uint32_t* counters, float scale) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	counters[i] = uint32_t(roundf(counters[i] * scale));
}

__global__ void spline_assign_float(uint32_t n_elements, float value, float* __restrict__ out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	out[i] = value;
}

__global__ void spline_init_rays_with_payload_kernel_sdf(
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

__host__ __device__ uint32_t spline_sample_discrete(float uniform_sample, const float* __restrict__ cdf, int length) {
	return binary_search(uniform_sample, cdf, length);
}

__global__ void spline_sample_uniform_on_triangle_kernel(
	uint32_t n_elements, const float* __restrict__ cdf, uint32_t length, const Triangle* __restrict__ triangles, vec3* __restrict__ sampled_positions
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	vec3 sample = sampled_positions[i];
	uint32_t tri_idx = spline_sample_discrete(sample.x, cdf, length);

	sampled_positions[i] = triangles[tri_idx].sample_uniform_position(sample.yz());
}

void Testbed::render_spline(
	cudaStream_t stream,
	CudaDevice& device,
	const distance_fun_t& distance_function,
	const normals_fun_t& normals_function,
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
	auto* octree_ptr = m_sdf.uses_takikawa_encoding || m_sdf.use_triangle_octree ? m_sdf.triangle_octree.get() : nullptr;

	SphereTracer tracer;

	uint32_t n_octree_levels = octree_ptr ? octree_ptr->depth() : 0;

	BoundingBox sdf_bounding_box = m_aabb;
	sdf_bounding_box.inflate(m_sdf.zero_offset);

	if (m_jit_fusion) {
		if (!device.fused_render_kernel()) {
			try {
				device.set_fused_render_kernel(
					std::make_unique<CudaRtcKernel>(
						"trace_sdf",
						fmt::format(
							"{}\n#include <neural-graphics-primitives/fused_kernels/trace_sdf.cuh>\n",
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

	bool gt_raytrace = m_render_ground_truth && m_sdf.groundtruth_mode == ESDFGroundTruthMode::RaytracedMesh;

	auto trace = [&](SphereTracer& tracer) {
		if (gt_raytrace) {
			return tracer.trace_bvh(m_sdf.triangle_bvh.get(), m_sdf.triangles_gpu.data(), stream);
		} else {
			return tracer.trace(
				distance_function,
				m_network.get(),
				m_sdf.zero_offset,
				m_sdf.distance_scale,
				m_sdf.maximum_distance,
				sdf_bounding_box,
				get_floor_y(),
				octree_ptr,
				n_octree_levels,
				stream
			);
		}
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
		if (m_sdf.analytic_normals || gt_raytrace) {
			normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);
		} else {
			float fd_normals_epsilon = m_sdf.fd_normals_epsilon;

			FiniteDifferenceNormalsApproximator fd_normals;
			fd_normals.normal(n_hit, distance_function, rays_hit.pos, rays_hit.normal, fd_normals_epsilon, stream);
		}

		if (render_mode == ERenderMode::Shade && n_hit > 0) {
			// Shadow rays towards the sun
			SphereTracer shadow_tracer;

			shadow_tracer.init_rays_from_data(n_hit, rays_hit, stream);
			shadow_tracer.set_fused_trace_kernel(tracer.fused_trace_kernel());
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_sdf.shadow_sharpness);
			RaysSdfSoa& shadow_rays_init = shadow_tracer.rays_init();
			linear_kernel(
				spline_prepare_shadow_rays,
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
				spline_write_shadow_ray_result,
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
		spline_shade_kernel_sdf,
		0,
		stream,
		n_hit,
		m_aabb,
		get_floor_y(),
		render_mode,
		m_sdf.brdf,
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

void Testbed::load_spline(const fs::path& data_path) {
	tlog::info() << "Trying to load spline at: " << data_path;
	std::ifstream file(data_path.str());
	if (!file.is_open()) {
		tlog::error() << "Could not open spline .bezdat file: " << data_path;
		return;
	}

	std::string line;

	if (!std::getline(file, line)) {
		tlog::error() << "Spline .bezdat file is empty.";
		return;
	}



	std::stringstream ss(line);

	while(std::getline(file, line)) {
		if(line.empty()) continue;

		std::istringstream iss(line);
		std::string tag;
		iss >> tag;

		if(tag == "PT")
		{
			Spline::Point p;
			iss >> p.pos[0] >> p.pos[1] >> p.pos[2]
				>> p.radius
				>> p.color[0] >> p.color[1] >> p.color[2];
			m_spline_sdf.points.push_back(p);
		} else if(tag == "BC") {
			Spline::Segment seg;
			int idx;

			while(iss >> idx) {
				seg.indices.push_back(idx);
			}

			if(!seg.indices.empty()) {
				m_spline_sdf.segments.push_back(std::move(seg));
			}
		}
	}

	//
	//
	//		load spline
	//
	//
	//
	// tlog::success() << "Loaded sphere: center = (" << x << ", " << y << ", " << z << "), radius = " << r;
	//
	// m_raw_aabb.min = vec3(std::numeric_limits<float>::infinity());
	// m_raw_aabb.max = vec3(-std::numeric_limits<float>::infinity());
	//
	// m_raw_aabb.enlarge(m_sphere.position + vec3(m_sphere.radius));
	// m_raw_aabb.enlarge(m_sphere.position - vec3(m_sphere.radius));
	//
	// const float inflation = 0.005f;
	//
	// m_raw_aabb.inflate(length(m_raw_aabb.diag()) * inflation);
	// float scale = compMax(m_raw_aabb.diag());
	//
	// // Normalize sphere coords to lie withing [0,1]^3
	// m_sphere.position = (m_sphere.position - m_raw_aabb.min - 0.5f * m_raw_aabb.diag()) / scale + vec3(0.5f);
	//
	// m_aabb = {};
	//
	// m_aabb.enlarge(m_sphere.position + vec3(m_sphere.radius));
	// m_aabb.enlarge(m_sphere.position - vec3(m_sphere.radius));
	//
	// m_render_aabb = m_aabb;
	// m_render_aabb_to_local = mat3(1.0f);
}

void Testbed::generate_training_samples_spline(vec3* positions, float* distances, uint32_t n_to_generate, cudaStream_t stream, bool uniform_only) {
	uint32_t n_to_generate_base = n_to_generate / 8;
	const uint32_t n_to_generate_surface_exact = uniform_only ? 0 : n_to_generate_base * 4;
	const uint32_t n_to_generate_surface_offset = uniform_only ? 0 : n_to_generate_base * 3;
	const uint32_t n_to_generate_uniform = uniform_only ? n_to_generate : n_to_generate_base * 1;

	const uint32_t n_to_generate_surface = n_to_generate_surface_exact + n_to_generate_surface_offset;

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void Testbed::train_spline(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 1;
	const uint32_t n_input_dims = 3;

	if (m_spline_sdf.training.size >= target_batch_size) {
		// Auxiliary matrices for training
		const uint32_t batch_size = (uint32_t)std::min(m_spline_sdf.training.size, target_batch_size);

		// Permute all training records to de-correlate training data
		linear_kernel(
			shuffle<vec3>,
			0,
			stream,
			m_spline_sdf.training.size,
			1,
			m_training_step,
			m_spline_sdf.training.positions.data(),
			m_spline_sdf.training.positions_shuffled.data()
		);
		linear_kernel(
			shuffle<float>,
			0,
			stream,
			m_spline_sdf.training.size,
			1,
			m_training_step,
			m_spline_sdf.training.distances.data(),
			m_spline_sdf.training.distances_shuffled.data()
		);

		GPUMatrix<float> training_target_matrix(m_spline_sdf.training.distances_shuffled.data(), n_output_dims, batch_size);
		GPUMatrix<float> training_batch_matrix((float*)(m_spline_sdf.training.positions_shuffled.data()), n_input_dims, batch_size);

		auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

		m_training_step++;

		if (get_loss_scalar) {
			m_loss_scalar.update(m_trainer->loss(stream, *ctx));
		}
	}
}

void Testbed::training_prep_spline(uint32_t batch_size, cudaStream_t stream) {
	if (m_spline_sdf.training.generate_sdf_data_online) {
		m_spline_sdf.training.size = batch_size;
		m_spline_sdf.training.positions.enlarge(m_spline_sdf.training.size);
		m_spline_sdf.training.positions_shuffled.enlarge(m_spline_sdf.training.size);
		m_spline_sdf.training.distances.enlarge(m_spline_sdf.training.size);
		m_spline_sdf.training.distances_shuffled.enlarge(m_spline_sdf.training.size);

		generate_training_samples_spline(
			m_spline_sdf.training.positions.data(), m_spline_sdf.training.distances.data(), batch_size, stream, false
		);
	}
}

} // namespace ngp
