#type compute
#version 450
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D depthTexture; 
layout(binding = 1) uniform samplerCube irradianceMap;
layout(binding = 2) uniform samplerCube prefilterMap;
layout(binding = 3) uniform sampler2D brdfLUT;
layout(binding = 4) uniform sampler2D ssaoTexture;

layout(rgba8, binding = 0) readonly uniform image2D diffuseImage;
layout(rg16f, binding = 1) readonly uniform image2D normalImage;
layout(rg8, binding = 2) readonly uniform image2D rougnessMetalnessImage;
layout(rgba16f, binding = 3) writeonly uniform image2D lightOutput;
layout(rgba16f, binding = 4) writeonly uniform image2D brightPassOutput;

#define DEBUG
#ifdef DEBUG
	layout(r32f, binding = 5) uniform image2D lightCountImage;
#endif

shared uint minDepth;
shared uint maxDepth;
shared uint tileDepthMask;
shared uint pointLightCount;
shared uint spotLightCount;
shared uint lightProbesCount;
shared uint pixelsToShade;
shared uint startIndex;

#define MAX_WORK_GROUP_SIZE 16
#define THREADS_PER_TILE MAX_WORK_GROUP_SIZE * MAX_WORK_GROUP_SIZE
#define M_PI 3.14159265359 

layout (std140) uniform Camera
{
	vec4 pos;
	mat4 view;
	mat4 projection;
	mat4 invertedView;
	mat4 invertedProjection;
} camera;

struct DirLight {
	vec3 direction;
	float nothing;
	vec3 color;
	float nothing2;
};

struct PointLight {
	vec4 positionAndRadius; // radius in w component
	vec3 color;
	float nothing2;
	mat4 modelMat;
};

struct SpotLight {
	vec3 position;
	float cutOff;
	vec3 color;
	float outerCutOff;
	vec3 direction;
	float maxDistance;
	vec4 boundingSphere; //xyz - sphere center, w - radius 
};

struct LightProbe {
	int64_t irradianceCubemapHandle;
	int64_t prefilterCubemapHandle;
	vec3 position;
	float radius;
};

layout(std430) buffer DirLightData
{
	DirLight dirLights[];
};

layout(std430) buffer PointLightData
{
	PointLight pointLights[];
};

layout(std430) buffer SpotLightData
{
	SpotLight spotLights[];
};

layout(std430) buffer LightProbeData
{
	LightProbe lightProbes[];
};

layout(std430) buffer PointLightIndices
{
	uint pointLightIndices[];
};

layout(std430) buffer SpotLightIndices
{
	uint spotLightIndices[];
};

//layout(std430) buffer LightProbeIndices
//{
//	uint lightProbeIndices[];
//};

struct Material
{
	vec3 albedo;
	float roughness;
	float metalness;
	vec3 F0;
};

struct AABB 
{
	vec3 aabbCenter;
	vec3 aabbHalfSize;
};

struct Frustum
{
	vec4 planes[6];
};

//light culling functions
AABB createTileAABB(float minDepthZ, float maxDepthZ, vec2 texSize);
Frustum createTileFrustum(float minDepthZ, float maxDepthZ, vec2 texSize);
bool testSphereVsAABB(vec3 sphereCenter, float sphereRadius, vec3 AABBCenter, vec3 AABBHalfSize);
vec3 fromNdcToViewSpace(vec4 ndcPoint);
float pixToNDC(uint point, float dimSize);
vec3 createPlane(vec3 p1, vec3 p2);
vec4 createPlanePerpendicularToView(vec3 dir1, vec3 dir2, vec3 pointOnPlane);
float getDistanceFromPlane(vec3 position, vec3 plane);
float getDistanceFromPerpendicularPlane(vec3 position, vec4 perpendicularPlane);
void cullPointLights(AABB tileAABB, float minDepthVS, float depthRangeRecip);
void cullPointLights(Frustum tileFrustum, float minDepthVS, float depthRangeRecip);
void cullSpotLights(AABB tileAABB);
//void cullLightProbes(AABB tileAABB, float minDepthVS, float depthRangeRecip);
//

vec3 worldPosFromDepth(float depth, vec2 texCoords);
vec3 decodeViewSpaceNormal(vec2 enc);
vec3 getBrightPassColor(vec3 color);

float normalDistributionGGX(vec3 N, vec3 H, float roughness);
vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0);
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);
float geometrySchlickGGX(float cosTheta, float k);
float geometrySmith(float NdotL, float NdotV, float roughness);
float computeSpecOcclusion(float NdotV, float AO, float roughness);
float calculateAttenuation(vec3 lightPos, vec3 Pos);
float calculateAttenuation(vec3 lightPos, vec3 pos, float maxDistance);

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);

void main()
{
	vec2 texSize = vec2(imageSize(diffuseImage));
	ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);

	if (gl_LocalInvocationIndex == 0)
	{
		pointLightCount = 0;
		spotLightCount = 0;
		lightProbesCount = 0;
		pixelsToShade = 0;
		minDepth = 0xffffffffu;
		maxDepth = 0;
		tileDepthMask = 0;

		uint nrOfElementsInColum = 256 * gl_NumWorkGroups.y;
		startIndex = nrOfElementsInColum * gl_WorkGroupID.x + 256 * gl_WorkGroupID.y;
	}

	barrier();

	float depthFloat = texelFetch(depthTexture, texCoords, 0).x;

	if (depthFloat != 0.0f)
	{
		atomicAdd(pixelsToShade, 1);
	}

	barrier();

	if (pixelsToShade == 0)
	{
		return;
	}
	
	uint depthInt = uint(depthFloat * 0xffffffffu);

	//Calcutate the depth range of this tile
	//Depth values are reversed, that means that zFar = 0, zNear = 1
	atomicMin(minDepth, depthInt); //tile zFar
	atomicMax(maxDepth, depthInt); //tile zNear

	// wait for process all invocations
	barrier();

	//convert min and max depth back to the float
	float maxDepthZ = float(float(maxDepth) / float(0xffffffffu));
	float minDepthZ = float(float(minDepth) / float(0xffffffffu));

	float minDepthVS = fromNdcToViewSpace(vec4(0.0f, 0.0f, max(minDepthZ, 0.00001f), 1.0f)).z;
	float maxDepthVS = fromNdcToViewSpace(vec4(0.0f, 0.0f, maxDepthZ, 1.0f)).z;
	float pixelDepthVS = fromNdcToViewSpace(vec4(0.0f, 0.0f, depthFloat, 1.0f)).z;

	//assigning linear depth [0,32] for each pixel from tile view space depth range
	float depthRangeRecip = 32.0f / (maxDepthVS - minDepthVS); // positive result
	//calculating proper bit index  
	uint depthMaskCellIndex = uint(max(0, min(32, floor((pixelDepthVS - minDepthVS) * depthRangeRecip))));

	atomicOr(tileDepthMask, 1 << depthMaskCellIndex);

	barrier();

	AABB tileAABB = createTileAABB(minDepthZ, maxDepthZ, texSize);
	//Frustum tileFrustum = createTileFrustum(minDepthZ, maxDepthZ, texSize);
	barrier();

	cullPointLights(tileAABB, minDepthVS, depthRangeRecip);
	//cullPointLights(tileFrustum, minDepthVS, depthRangeRecip);
	cullSpotLights(tileAABB);
	//cullLightProbes(tileAABB, minDepthVS, depthRangeRecip);
	
	barrier();

#ifdef DEBUG
	if (gl_LocalInvocationIndex == 0)
	{
		imageStore(lightCountImage, ivec2(gl_WorkGroupID.xy), vec4(float(pointLightCount)));
	}
#endif

	if (depthFloat == 0)
		return;

//light calculations in world space
	vec3 albedo = imageLoad(diffuseImage, texCoords).rgb;
	vec2 encodedNormal = imageLoad(normalImage, texCoords).xy;
	vec2 roughnessMetalness = imageLoad(rougnessMetalnessImage, texCoords).xy;
	

	Material material = {
		albedo,
		roughnessMetalness.x,
		roughnessMetalness.y,
		vec3(0)
	};

	vec2 screenSpaceTexCoords = vec2(texCoords) / texSize;
	vec3 P = worldPosFromDepth(depthFloat, screenSpaceTexCoords);
	vec3 N = (camera.invertedView * vec4(decodeViewSpaceNormal(encodedNormal), 0.0f)).xyz;
	vec3 V = normalize(camera.pos.xyz - P);

	//vec3 F0 = vec3(0.04);
	vec3 F0 = vec3(0.16) * pow(material.roughness, 2); //frostbite3 fresnel reflectance https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf page 14
	material.F0 = mix(F0, material.albedo, material.metalness);

	vec3 L0 = directionalLightAddition(V, N, material);
	L0 += pointLightAddition(V, N, P, material);
	L0 += spotLightAddition(V, N, P, material);

//IBL here
	vec3 ambient = vec3(0.0);
	float ssao = texture(ssaoTexture, screenSpaceTexCoords).x;
	if (lightProbes.length() == 0)
	{
		float NdotV = max(dot(N, V), 0.0);
		vec3 kS = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
		vec3 kD = 1.0 - kS;
		kD *= 1.0 - material.metalness;
		vec3 irradiance = texture(irradianceMap, N).rgb;
		vec3 diffuse    = irradiance * material.albedo;

		vec3 R = reflect(-V, N);
		vec3 F = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
		const float MAX_REFLECTION_LOD = 4.0;

		//float mipMapLevel = material.roughness * MAX_REFLECTION_LOD; //base
		float mipMapLevel = sqrt(material.roughness * MAX_REFLECTION_LOD); //frostbite 3
		vec3 prefilteredColor = textureLod(prefilterMap, R, mipMapLevel).rgb;    
		vec2 brdf = texture(brdfLUT, vec2(NdotV, material.roughness)).rg;
		
		float specOcclusion = computeSpecOcclusion(NdotV, ssao, material.roughness);
		vec3 specular = prefilteredColor * (F * brdf.x + brdf.y) * specOcclusion;
		
		ambient = (kD * diffuse + specular);
	}
	else
	{
		for(int i = 0; i < lightProbes.length(); ++i)
		{
			LightProbe lightProbe = lightProbes[i];
			samplerCube irradianceSampler = samplerCube(lightProbe.irradianceCubemapHandle);
			samplerCube prefilterSampler = samplerCube(lightProbe.prefilterCubemapHandle);
			
			float NdotV = max(dot(N, V), 0.0);
			vec3 kS = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
			vec3 kD = 1.0 - kS;
			kD *= 1.0 - material.metalness;
			vec3 irradiance = texture(irradianceSampler, N).rgb;
			vec3 diffuse    = irradiance * material.albedo;

			vec3 R = reflect(-V, N);
			vec3 F = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
			const float MAX_REFLECTION_LOD = 4.0;

			//float mipMapLevel = material.roughness * MAX_REFLECTION_LOD; //base
			float mipMapLevel = sqrt(material.roughness * MAX_REFLECTION_LOD); //frostbite 3
			vec3 prefilteredColor = textureLod(prefilterSampler, R, mipMapLevel).rgb;    
			vec2 brdf = texture(brdfLUT, vec2(NdotV, material.roughness)).rg;
		
			float specOcclusion = computeSpecOcclusion(NdotV, ssao, material.roughness);
			vec3 specular = prefilteredColor * (F * brdf.x + brdf.y) * specOcclusion;
		
			ambient = (kD * diffuse + specular);
		}
//		for(int i = 0; i < lightProbesCount; ++i)
//		{
//			LightProbe lightProbe = lightProbes[lightProbeIndices[i]];
//			samplerCube irradianceSampler = samplerCube(lightProbe.irradianceCubemapHandle);
//			samplerCube prefilterSampler = samplerCube(lightProbe.prefilterCubemapHandle);
//			
//			float NdotV = max(dot(N, V), 0.0);
//			vec3 kS = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
//			vec3 kD = 1.0 - kS;
//			kD *= 1.0 - material.metalness;
//			vec3 irradiance = texture(irradianceSampler, N).rgb;
//			vec3 diffuse    = irradiance * material.albedo;
//
//			vec3 R = reflect(-V, N);
//			vec3 F = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
//			const float MAX_REFLECTION_LOD = 4.0;
//
//			//float mipMapLevel = material.roughness * MAX_REFLECTION_LOD; //base
//			float mipMapLevel = sqrt(material.roughness * MAX_REFLECTION_LOD); //frostbite 3
//			vec3 prefilteredColor = textureLod(prefilterSampler, R, mipMapLevel).rgb;    
//			vec2 brdf = texture(brdfLUT, vec2(NdotV, material.roughness)).rg;
//		
//			float specOcclusion = computeSpecOcclusion(NdotV, ssao, material.roughness);
//			vec3 specular = prefilteredColor * (F * brdf.x + brdf.y) * specOcclusion;
//		
//			ambient = (kD * diffuse + specular);
//		}
	}
	
	vec4 color = vec4(L0 + ambient, 1);// * ssao;
	
	bvec4 valid = isnan(color);
	if ( valid.x || valid.y || valid.z || valid.w )
	{
		color = vec4(0.5f);
	}

	barrier();
	//imageStore(lightOutput, texCoords, vec4(pointLightCount, spotLightCount, 0, 0));
	imageStore(lightOutput, texCoords, color);
	imageStore(brightPassOutput, texCoords, vec4(getBrightPassColor(color.xyz), 1.0f));
}

Frustum createTileFrustum(float minDepthZ, float maxDepthZ, vec2 texSize)
{
	//calculate tile corners (2d points in pixels)
	uint minX = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.x;
	uint minY = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.y;
	uint maxX = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.x + 1);
	uint maxY = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.y + 1);

	//Convert tile corners to NDC and then to view space
	vec3 tileCorners[4];
	const float infZProjectionAdjustment = 0.00001f; //protection when projection far plane z = 0 is inf
	const float depth = max(minDepthZ, infZProjectionAdjustment);
	tileCorners[0] = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), depth, 1.0f));
	tileCorners[1] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(minY, texSize.y), depth, 1.0f));
	tileCorners[2] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), depth, 1.0f));
	tileCorners[3] = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(maxY, texSize.y), depth, 1.0f));

	//create the frustum planes by using the product between these points
	vec4 planes[6];
	for(int i = 0; i < 4; ++i)
	{
		planes[i].xyz = createPlane(tileCorners[i],tileCorners[(i+1) & 3]);
	}

	//create far plane
	planes[4] = createPlanePerpendicularToView(tileCorners[0] - tileCorners[1], 
		tileCorners[2] - tileCorners[1], tileCorners[1]);

	//create nearPlane
	vec3 nearTileCorners[3];
	nearTileCorners[0] = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), maxDepthZ, 1.0f));
	nearTileCorners[1] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(minY, texSize.y), maxDepthZ, 1.0f));
	nearTileCorners[2] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), maxDepthZ, 1.0f));

	planes[5] = createPlanePerpendicularToView(nearTileCorners[2] - nearTileCorners[1], 
		nearTileCorners[0] - nearTileCorners[1], nearTileCorners[1]);

	return Frustum(planes);
}

AABB createTileAABB(float minDepthZ, float maxDepthZ, vec2 texSize)
{
	//calculate tile corners (2d points in pixels)
	uint minX = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.x;
	uint minY = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.y;
	uint maxX = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.x + 1);
	uint maxY = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.y + 1);

	//Convert bottom tile corners to NDC and then to view space
	const float depth = max(minDepthZ, 0.00001f); //protection when projection far plane z = 0 is inf
	vec3 tileFarBottomLeftCorner = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), depth, 1.0f));
	vec3 tileFarUpperRightCorner = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), depth, 1.0f));

	vec3 tileNearBottomLeftCorner = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), maxDepthZ, 1.0f));
	vec3 tileNearUpperRightCorner = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), maxDepthZ, 1.0f));

	//AABB's min-max points
	float minXViewSpace = min(tileFarBottomLeftCorner.x, tileNearBottomLeftCorner.x);
	float minYViewSpace = min(tileFarBottomLeftCorner.y, tileNearBottomLeftCorner.y);
	vec3 minPoint = vec3(minXViewSpace, minYViewSpace, tileFarBottomLeftCorner.z);

	float maxXViewSpace = max(tileFarUpperRightCorner.x, tileNearUpperRightCorner.x);
	float maxYViewSpace = max(tileFarUpperRightCorner.y, tileNearUpperRightCorner.y);
	vec3 maxPoint = vec3(maxXViewSpace, maxYViewSpace, tileNearUpperRightCorner.z); 

	vec3 aabbHalfSize = abs(maxPoint - minPoint) * 0.5;
	vec3 aabbCenter = minPoint + aabbHalfSize;
	
	return AABB(aabbCenter, aabbHalfSize);
}

bool testSphereVsAABB(vec3 sphereCenter, float sphereRadius, vec3 AABBCenter, vec3 AABBHalfSize)
{
	vec3 delta = vec3(0);
	delta.x = max(0, abs(AABBCenter.x - sphereCenter.x) - AABBHalfSize.x);
	delta.y = max(0, abs(AABBCenter.y - sphereCenter.y) - AABBHalfSize.y);
	delta.z = max(0, abs(AABBCenter.z - sphereCenter.z) - AABBHalfSize.z);
	float distSq = dot(delta, delta);
	return distSq <= (sphereRadius * sphereRadius);
}

vec3 fromNdcToViewSpace(vec4 ndcPoint)
{
	vec4 viewSpacePosition = camera.invertedProjection * ndcPoint;
	return viewSpacePosition.xyz / viewSpacePosition.w;
}

float pixToNDC(uint point, float dimSize)
{
	return (float(point) / dimSize) * 2.0f - 1.0f; 
}

vec3 createPlane(vec3 p1, vec3 p2)
{
	//coefficients for plane equation are:
	//A = normalize(cross(p1, p2)).x;
	//B = normalize(cross(p1, p2)).y;
	//C = normalize(cross(p1, p2)).z;
	//D = 0 -> because plane goes through the point (0,0,0)
	return normalize(cross(p1, p2));
}

vec4 createPlanePerpendicularToView(vec3 dir1, vec3 dir2, vec3 pointOnPlane)
{
	vec4 planeCoeffs = {0,0,0,0};
	planeCoeffs.xyz = normalize(cross(dir1, dir2));
	planeCoeffs.w = -dot(planeCoeffs.xyz, pointOnPlane);
	return planeCoeffs;
}

float getDistanceFromPlane(vec3 position, vec3 plane)
{
	return dot(position, plane);
}

float getDistanceFromPerpendicularPlane(vec3 position, vec4 perpendicularPlane)
{
	return dot(position, perpendicularPlane.xyz) + perpendicularPlane.w;
}

void cullPointLights(Frustum tileFrustum, float minDepthVS, float depthRangeRecip)
{
	//Now check the lights against the frustum and append them to the list

	for (uint i = gl_LocalInvocationIndex; i < pointLights.length(); i += THREADS_PER_TILE)
	{
		uint il = i;
		PointLight p = pointLights[il];

		vec3 lightViewPosition = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
		float r = p.positionAndRadius.w;

		float zMin = lightViewPosition.z - r;
		float zMax = lightViewPosition.z + r;

		uint lightMaskCellIndexStart = uint(max(0, min(32, floor(zMin - minDepthVS) * depthRangeRecip)));
		uint lightMaskCellIndexEnd = uint(max(0, min(32, floor(zMax - minDepthVS) * depthRangeRecip)));

		uint lightMask = 0xffffffffu;
		lightMask >>= 31 - (lightMaskCellIndexEnd - lightMaskCellIndexStart);
		lightMask <<= lightMaskCellIndexStart;

		bool intersect2_5D = bool(tileDepthMask & lightMask);

		if (!intersect2_5D)
			continue;

		if( 
			( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[0].xyz) < r ) &&
			( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[1].xyz) < r ) &&
			( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[2].xyz) < r ) &&
			( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[3].xyz) < r)  &&
			( getDistanceFromPerpendicularPlane(lightViewPosition, tileFrustum.planes[4]) < r) &&
			( getDistanceFromPerpendicularPlane(lightViewPosition, tileFrustum.planes[5]) < r)
			)
		{
			uint id = atomicAdd(pointLightCount, 1);
			pointLightIndices[startIndex + id] = il;
		}
	}
}

void cullPointLights(AABB tileAABB, float minDepthVS, float depthRangeRecip)
{
	//Now check the lights against the frustum and append them to the list

	for (uint i = gl_LocalInvocationIndex; i < pointLights.length(); i += THREADS_PER_TILE)
	{
		uint il = i;
		PointLight p = pointLights[il];

		vec3 lightViewPosition = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
		float r = p.positionAndRadius.w;

		float zMin = lightViewPosition.z - r;
		float zMax = lightViewPosition.z + r;

		uint lightMaskCellIndexStart = uint(max(0, min(32, floor(zMin - minDepthVS) * depthRangeRecip)));
		uint lightMaskCellIndexEnd = uint(max(0, min(32, floor(zMax - minDepthVS) * depthRangeRecip)));

		uint lightMask = 0xffffffffu;
		lightMask >>= 31 - (lightMaskCellIndexEnd - lightMaskCellIndexStart);
		lightMask <<= lightMaskCellIndexStart;

		bool intersect2_5D = bool(tileDepthMask & lightMask);

		if (!intersect2_5D)
			continue;

		if( testSphereVsAABB(lightViewPosition, r, tileAABB.aabbCenter, tileAABB.aabbHalfSize) )
		{
			uint id = atomicAdd(pointLightCount, 1);
			pointLightIndices[startIndex + id] = il;
		}
	}
}

void cullSpotLights(AABB tileAABB)
{
	//Now check the lights against the frustum and append them to the list

	for (uint i = gl_LocalInvocationIndex; i < spotLights.length(); i += THREADS_PER_TILE)
	{
		uint il = i;
		SpotLight s = spotLights[il];

		vec3 lightViewPosition = (camera.view * vec4(s.boundingSphere.xyz, 1.0f)).xyz;
		float r = s.boundingSphere.w;

		if( testSphereVsAABB(lightViewPosition, r, tileAABB.aabbCenter, tileAABB.aabbHalfSize) )
		{
			uint id = atomicAdd(spotLightCount, 1);
			spotLightIndices[startIndex + id] = il;
		}
	}
}

//void cullLightProbes(AABB tileAABB, float minDepthVS, float depthRangeRecip)
//{
//	//Now check the lights against the frustum and append them to the list
//
//	for (uint i = gl_LocalInvocationIndex; i < lightProbes.length(); i += THREADS_PER_TILE)
//	{
//		LightProbe lightProbe = lightProbes[i];
//
//		vec3 lightViewPosition = (camera.view * vec4(lightProbe.position, 1.0f)).xyz;
//		float r = lightProbe.radius;
//
//		float zMin = lightViewPosition.z - r;
//		float zMax = lightViewPosition.z + r;
//
//		uint lightMaskCellIndexStart = uint(max(0, min(32, floor(zMin - minDepthVS) * depthRangeRecip)));
//		uint lightMaskCellIndexEnd = uint(max(0, min(32, floor(zMax - minDepthVS) * depthRangeRecip)));
//
//		uint lightMask = 0xffffffffu;
//		lightMask >>= 31 - (lightMaskCellIndexEnd - lightMaskCellIndexStart);
//		lightMask <<= lightMaskCellIndexStart;
//
//		bool intersect2_5D = bool(tileDepthMask & lightMask);
//
//		if (!intersect2_5D)
//			continue;
//
//		if( testSphereVsAABB(lightViewPosition, r, tileAABB.aabbCenter, tileAABB.aabbHalfSize) )
//		{
//			uint id = atomicAdd(lightProbesCount, 1);
//			lightProbeIndices[startIndex + id] = i;
//		}
//	}
//}

vec3 worldPosFromDepth(float depth, vec2 texCoords) {
    vec4 clipSpacePosition = vec4(texCoords * 2.0 - 1.0, depth, 1.0);
    vec4 viewSpacePosition = camera.invertedProjection * clipSpacePosition;
    vec4 worldSpacePosition = camera.invertedView * viewSpacePosition;
    return worldSpacePosition.xyz /= worldSpacePosition.w; //perspective division
}

vec3 decodeViewSpaceNormal(vec2 enc)
{
	//Lambert Azimuthal Equal-Area projection
	//http://aras-p.info/texts/CompactNormalStorage.html
	vec2 fenc = enc*4-2;
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}

vec3 getBrightPassColor(vec3 color)
{
    const float luma = dot(color, vec3(0.299, 0.587, 0.114));
    float weight = 1 / (1 + luma);
    return color * weight;
}

vec3 directionalLightAddition(vec3 V, vec3 N, Material m)
{
	float NdotV = max(dot(N, V), 0.0f);

	vec3 L0 = { 0, 0, 0 };
	for (uint i = 0; i < dirLights.length(); ++i)
	{
		vec3 L = normalize(-dirLights[i].direction);
		vec3 H = normalize(V + L);

		float NdotL = max(dot(N, L), 0.0f);

		vec3 F = fresnelSchlick(V, H, m.F0);
		float D = normalDistributionGGX(N, H, m.roughness);
		float G = geometrySmith(NdotL, NdotV, m.roughness);

		vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
		vec3 diffuseColor = kD * m.albedo / M_PI;

		vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);
		
		L0 += (diffuseColor + specularColor) * dirLights[i].color * NdotL;
	}
	return L0;
}

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
	float NdotV = max(dot(N, V), 0.0f);

	vec3 L0 = { 0, 0, 0 };
	
	//for (int index = 0; index < pointLights.length(); ++index)
	for (int index = 0; index < pointLightCount; ++index)
	{
		//PointLight p = pointLights[index];
		PointLight p = pointLights[pointLightIndices[startIndex + index]];
	
		vec3 lightPos = p.positionAndRadius.xyz;
		float lightRadius = p.positionAndRadius.w;
		vec3 L = normalize(lightPos - Pos);
		vec3 H = normalize(V + L);

		float NdotL = max(dot(N, L), 0.0f);

		vec3 F = fresnelSchlick(V, H, m.F0);
		float D = normalDistributionGGX(N, H, m.roughness);
		float G = geometrySmith(NdotV, NdotL, m.roughness);
		
		vec3 radiance = p.color * calculateAttenuation(lightPos, Pos, lightRadius);

		vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
		vec3 diffuseColor = kD * m.albedo / M_PI;

		vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

		L0 += (diffuseColor + specularColor) * radiance * NdotL;
	}

	return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
	float NdotV = max(dot(N, V), 0.0f);

	vec3 L0 = { 0, 0, 0 };
	for (int index = 0; index < spotLightCount; ++index)
	{
		//PointLight p = pointLights[index];
		SpotLight s = spotLights[spotLightIndices[startIndex + index]];
		vec3 directionToLight = normalize(-s.direction);
		vec3 L = normalize(s.position - Pos);

		float theta = dot(directionToLight, L);
		float epsilon = max(s.cutOff - s.outerCutOff, 0.0);
		float intensity = clamp((theta - s.outerCutOff) / epsilon, 0.0, 1.0);  

		vec3 H = normalize(V + L);

		float NdotL = max(dot(N, L), 0.0f);

		vec3 F = fresnelSchlick(V, H, m.F0);
		float D = normalDistributionGGX(N, H, m.roughness);
		float G = geometrySmith(NdotV, NdotL, m.roughness);
		
		vec3 radiance = s.color * calculateAttenuation(s.position, Pos, s.maxDistance);
		radiance *= intensity;

		vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
		vec3 diffuseColor = kD * m.albedo / M_PI;

		vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

		L0 += (diffuseColor + specularColor) * radiance * NdotL;
	}
	return L0;
}

float normalDistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	
	float nom = a2;
	float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
	return nom / (M_PI * denom * denom);
}

vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0)
{
	float cosTheta = max(dot(V, H), 0.0);
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float geometrySchlickGGX(float cosTheta, float k)
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

float geometrySmith(float NdotL, float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;
	return geometrySchlickGGX(NdotL, k) * geometrySchlickGGX(NdotV, k);
}

float computeSpecOcclusion(float NdotV, float AO, float roughness)
{
    //https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    //page 77
	return clamp(pow(NdotV + AO, exp2(-16.0f * roughness - 1.0f)) - 1.0f + AO, 0.0f, 1.0f);
}

float calculateAttenuation(vec3 lightPos, vec3 Pos)
{
	float distance    = length(lightPos - Pos);
	float attenuation = 1.0 / (distance * distance);
	return attenuation; 
}

float calculateAttenuation(vec3 lightPos, vec3 pos, float maxDistance)
{
    //https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    //page 31
	float distance    = length(lightPos - pos);
    float attenuation = max((1.0 / (distance * distance) * (1 - distance / maxDistance)), 0.0000001f);
    return attenuation; 
}