// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#include "common.cuh"

// #include <render_path_tracing.h>

struct DenoiseGBuffer {
	vec3f normal;
	vec3f tangent;
	vec3f color;
	vec3f position;
	uint32_t strand_index;
	bool hit;
	float radius;
};

struct LaunchParams {
	float4* accumBuffer;
	float4* denoiseBuffer;
	float4* averageBuffer;
	int accumId;

	OptixTraversableHandle world;

	DenoiseGBuffer* gBuffer;
	DenoiseGBuffer* fittedGuffer;


	bool hasEnvLight, envPdfSampling;
	float envScale, envRotPhi;
	int envWidth, envHeight;
	cudaTextureObject_t env;
	cudaTextureObject_t conditionalPdf, conditionalCdf;
	cudaTextureObject_t marginalPdf, marginalCdf;

	bool hasDirectionalLights;
	DirectionalLight* dLights;
	int num_dlights;

	int num_total_lights;

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;

	struct {
		vec3f sig_a;
		float beta_m, beta_n;
		float alpha;

		float R_G, TT_G, TRT_G, TRRT_G;
	} hairData;

	int pathV1, pathV2;
	bool MIS;

	// Properties of the scene
	vec3f maxBound, minBound;
	float sceneScale;

	float positionSigma;
	float tangentSigma;
	float colorSigma;
	float thetaRange;
	float radiusScale;
	float weightThreshold;
	float tangentThreshold;
	float fitDeltaThreshold;

	uint32_t debugMode;
	int targetSpp;

};

#define VisualizeResults 0
#define VisualizeNormal 1
#define VisualizeTangent 2
#define VisualizePosition 3
#define VisualizeBlack 4

__constant__ LaunchParams optixLaunchParams;