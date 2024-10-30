// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "path_tracing.cuh"
#include "utils.cuh"

#include "curve_utils.cuh"
#include "disney_hair.cuh"
#include "frostbite_anisotropic.cuh"

#include "optix_common.cuh"

#include <optix_device.h>

OPTIX_RAYGEN_PROGRAM(rayGenCam)()
{
    // ---------------------
    // Path Tracing
    // ---------------------
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;
    
    // Pseudo-random number generator
    LCGRand rng = get_rng(optixLaunchParams.accumId + 10007, make_uint2(pixelId.x, pixelId.y),
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));
    
    // Shoot camera ray
    vec2f pixelOffset = vec2f(lcg_randomf(rng), lcg_randomf(rng));
    if(optixLaunchParams.targetSpp == 1)
        pixelOffset = vec2f(0.5f,0.5f);
    const vec2f screen = (vec2f(pixelId) + pixelOffset) / vec2f(self.frameBufferSize);
    
    RadianceRay ray;
    ray.origin
        = optixLaunchParams.camera.pos;
    ray.direction
        = normalize(optixLaunchParams.camera.dir_00
            + screen.u * optixLaunchParams.camera.dir_du
            + screen.v * optixLaunchParams.camera.dir_dv);

    // printf("RayGen %d %d\n",pixelId.x,pixelId.y);
                    
    
    Interaction si;
    si.hit = false;
    si.wo = -1.f * ray.direction;
    si.wi = ray.direction;
    
    owl::traceRay(optixLaunchParams.world, ray, si);
    
    vec3f color(0.f, 0.f, 0.f);
    int v1Stop = optixLaunchParams.pathV1 - 1;
    int v2Stop = optixLaunchParams.pathV2 - 1;
    
    if (si.hit == false) {
        color = si.Le;
        optixLaunchParams.gBuffer[fbOfs] = DenoiseGBuffer();
        optixLaunchParams.fittedGuffer[fbOfs] = DenoiseGBuffer();
    }
    else {
        color = pathTrace(si, rng, v1Stop, v2Stop);
        uint32_t accumId = optixLaunchParams.accumId;
        // if(accumId == 0) {
            optixLaunchParams.gBuffer[fbOfs].normal = si.n;
            optixLaunchParams.gBuffer[fbOfs].tangent = si.t;
            optixLaunchParams.gBuffer[fbOfs].radius = si.hair.radius;
            optixLaunchParams.gBuffer[fbOfs].color = si.color;
            optixLaunchParams.gBuffer[fbOfs].position = si.p;
            optixLaunchParams.gBuffer[fbOfs].strand_index = si.hair.hair_strand_index;
            optixLaunchParams.gBuffer[fbOfs].hit = true;
        // }
        // else {
        //     optixLaunchParams.gBuffer[fbOfs].normal = (optixLaunchParams.gBuffer[fbOfs].normal * accumId + si.n) / (accumId + 1);
        //     
        // }

      //  printf("strand %d normal %f %f %f tangent %f %f %f\n",si.hair.hair_strand_index,si.n.x,si.n.y,si.n.z,si.t.x,si.t.y,si.t.z);
        // optixLaunchParams.denoiseGBuffer[fbOfs].strand_index = si.hair.
    }

    if(optixLaunchParams.debugMode == VisualizeNormal) {
        color = 0.5f * si.n + 0.5f;
    }
    writePixel(color, optixLaunchParams.accumId,
        self.frameBuffer,
        optixLaunchParams.accumBuffer,
        optixLaunchParams.averageBuffer,
        fbOfs);
    optixLaunchParams.denoiseBuffer[fbOfs] = make_float4(color.x,color.y,color.z,1.0f);
    if(optixLaunchParams.debugMode == VisualizeNormal) {
        optixLaunchParams.gBuffer[fbOfs].normal.x = optixLaunchParams.averageBuffer[fbOfs].x *2 -1;
        optixLaunchParams.gBuffer[fbOfs].normal.y = optixLaunchParams.averageBuffer[fbOfs].y * 2-1;
        optixLaunchParams.gBuffer[fbOfs].normal.z = optixLaunchParams.averageBuffer[fbOfs].z * 2 -1;
    }
    
}

__device__
double Distance(int x0, int y0, int x1, int y1) {
    float d = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);
    // printf("Distance %d %d %d %d %f\n",x0,y0,x1,y1,d);
    return sqrtf((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}


__device__
vec3f ComputeSphereCenter(vec3f position,vec3f normal,vec3f tangent,float radius) {
    // return position;
    vec3f center = position + radius * normal;
    return center;
}

__device__ bool AlmostZero(float v) {
    return abs(v) < 1e-6f;
}

__device__
vec2i ComputeScreenSpace(vec3f position,vec3f dir_00,vec3f dir_du,vec3f dir_dv,vec3f center,vec2i frameBufferSize) {
    vec3f dir = center - position;
    dir = normalize(dir);
    float t = dot(dir,dir_00) / dot(dir_00,dir_00);
    float u = dot(dir,dir_du) / dot(dir_du,dir_du) / t;
    float v = dot(dir,dir_dv) / dot(dir_dv,dir_dv) / t;

    // if( AlmostZero(u) || AlmostZero(v)) {
    //     printf("dir %f %f %f dir_00 %f %f %f dir_du %f %f %f dir_dv %f %f %f\n",dir.x,dir.y,dir.z,dir_00.x,dir_00.y,dir_00.z,dir_du.x,dir_du.y,dir_du.z,dir_dv.x,dir_dv.y,dir_dv.z);
    //     printf("center position %f %f %f %f %f %f %f %f %f\n",center.x,center.y,center.z,position.x,position.y,position.z,dir.x,dir.y,dir.z);
    // }

    u+=0.5f;
    v+=0.5f;
    
    int centerx = int(u * frameBufferSize.x);
    int centery = int(v * frameBufferSize.y);
    return vec2i(centerx,centery);
}

OPTIX_RAYGEN_PROGRAM(fitGBuffer)(){
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;
    int leftx = max(0, pixelId.x - 1);
    int rightx = min(self.frameBufferSize.x - 1, pixelId.x + 1);
    int upy = max(0, pixelId.y - 1);
    int downy = min(self.frameBufferSize.y - 1, pixelId.y + 1);


  //  printf("Fitting pixel %d %d\n",pixelId.x,pixelId.y);
    
    if(optixLaunchParams.gBuffer[fbOfs].hit) {
        optixLaunchParams.fittedGuffer[fbOfs] = optixLaunchParams.gBuffer[fbOfs];
        return;
    }
    // printf("Fitting pixel %d %d\n",pixelId.x,pixelId.y);
    
    float minDistance = 1e20f;
    int m;
    int n;

    int validNeghborCount = 0;
    
    for (int i = leftx; i <= rightx; i++) {
        for (int j = upy; j <= downy; j++) {
            if(i == pixelId.x && j == pixelId.y) {
                continue;
            }
            int ofs = i + self.frameBufferSize.x * j;
            if(!optixLaunchParams.gBuffer[ofs].hit) {
                continue;
            }
            
            vec3f center = ComputeSphereCenter(optixLaunchParams.gBuffer[ofs].position,optixLaunchParams.gBuffer[ofs].normal,optixLaunchParams.gBuffer[ofs].tangent,optixLaunchParams.gBuffer[ofs].radius * optixLaunchParams.radiusScale);

            // vec3f dir = center - optixLaunchParams.camera.pos;
            // //
            vec3f dir_00 = optixLaunchParams.camera.dir_00 + 0.5f * (optixLaunchParams.camera.dir_du + optixLaunchParams.camera.dir_dv);
            // dir = normalize(dir);
            // float t = dot(dir,dir_00) / dot(dir_00,dir_00);
            // float u = dot(dir,optixLaunchParams.camera.dir_du) / dot(optixLaunchParams.camera.dir_du,optixLaunchParams.camera.dir_du) / t;
            // float v = dot(dir,optixLaunchParams.camera.dir_dv) / dot(optixLaunchParams.camera.dir_dv,optixLaunchParams.camera.dir_dv) / t;
            //
            // u+=0.5f;
            // v+=0.5f;
            //
            // int centerx = int(u * self.frameBufferSize.x);
            // int centery = int(v * self.frameBufferSize.y);
            
            //
            vec2i center_screen_space = ComputeScreenSpace(optixLaunchParams.camera.pos,dir_00,optixLaunchParams.camera.dir_du,optixLaunchParams.camera.dir_dv,center,self.frameBufferSize);
            int centerx = center_screen_space.x;
            int centery = center_screen_space.y;
            
            float delta = abs(Distance(i,j,centerx,centery) - Distance(pixelId.x,pixelId.y,centerx,centery));

          // printf("cent21e %d %d %d %d %f radius %f delta %f\n",centerx,centery,i,j,Distance(i,j,centerx,centery),optixLaunchParams.gBuffer[ofs].radius,delta);

            // printf("%f\n" ,Distance(i,j,pixelId.x,pixelId.y));
            if(delta < optixLaunchParams.fitDeltaThreshold &&  Distance(i,j,centerx,centery) < minDistance) {
                minDistance = Distance(i,j,centerx,centery);
                // printf("minDistance %f\n",minDistance);
                m = i;
                n = j;
            }
        }
    }

    if(minDistance < 1e20f) {
        // printf("Fitted pixel %d %d to %d %d\n",pixelId.x,pixelId.y,m,n);
        int ofs = m + self.frameBufferSize.x * n;
        optixLaunchParams.fittedGuffer[fbOfs] = optixLaunchParams.gBuffer[ofs];
        optixLaunchParams.gBuffer[fbOfs] = optixLaunchParams.gBuffer[ofs];
        //optixLaunchParams.averageBuffer[fbOfs] = optixLaunchParams.averageBuffer[ofs];
    }
    else {
        optixLaunchParams.fittedGuffer[fbOfs].strand_index = -100;

    }
}

__device__ __host__

float Distance(vec3f p0, vec3f p1) {
    return length(p0 - p1);
}
__device__ __host__

float Distance2(vec3f p0, vec3f p1) {
    return dot(p0 - p1, p0 - p1);
}
__device__ __host__

float GetFilterWeight(DenoiseGBuffer g0, DenoiseGBuffer g1,vec3f color0,vec3f color1,vec2i pixelId0,vec2i pixelId1) {
    if(g0.strand_index != g1.strand_index && dot(g0.tangent,g1.tangent) < optixLaunchParams.tangentThreshold) {
       // printf("Different strand %d %d\n",g0.strand_index,g1.strand_index);
        return 0.0f;
    }
    float positionSigma = optixLaunchParams.positionSigma;
    float tangentSigma = optixLaunchParams.tangentSigma;
    float colorSigma = optixLaunchParams.colorSigma;

    float distancePixel = (pixelId0.x - pixelId1.x) * (pixelId0.x - pixelId1.x) + (pixelId0.y - pixelId1.y) * (pixelId0.y - pixelId1.y);
    
    float distancePos = exp(-distancePixel * positionSigma);
    float distanceTangent = exp(-Distance2(g0.tangent,g1.tangent) * tangentSigma);
    float distanceColor = exp(-Distance2(color0,color1) * colorSigma);


   //printf("Distance %f %f %f %f\n",distancePixel,distancePos,distanceTangent,distanceColor);

    // return 1;
    float weight = distancePos * distanceTangent * distanceColor;
    if(weight == 1.f && pixelId0!=pixelId1){
       printf("strand %d %d tangent %f %f %f %f %f %f distance %f %f %f %f\n",g0.strand_index,g1.strand_index,g0.tangent.x,g0.tangent.y,g0.tangent.z,g1.tangent.x,g1.tangent.y,g1.tangent.z,distancePixel,distancePos,distanceTangent,distanceColor);
    }
    return distancePos * distanceTangent * distanceColor;
}
__device__ __host__

float calculateTheta(const vec2f& center, const vec2f& point) {
    float dx = point.x - center.x;
    float dy = point.y - center.y;
    return std::atan2(dy, dx);
}

OPTIX_RAYGEN_PROGRAM(denoise)(){
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    if(!optixLaunchParams.gBuffer[fbOfs].hit) {
        optixLaunchParams.denoiseBuffer[fbOfs]  = optixLaunchParams.averageBuffer[fbOfs];
        // optixLaunchParams.denoiseBuffer[fbOfs]  = make_float4(1.f,0.f,0.f,1.f);
        writePixel(optixLaunchParams.denoiseBuffer,self.frameBuffer,fbOfs);
        return;
    }

    vec3f center = ComputeSphereCenter(optixLaunchParams.gBuffer[fbOfs].position,optixLaunchParams.gBuffer[fbOfs].normal,optixLaunchParams.gBuffer[fbOfs].tangent,optixLaunchParams.gBuffer[fbOfs].radius * optixLaunchParams.radiusScale);
    // vec2f ceneter_screen_space = vec2f(dot(center - optixLaunchParams.camera.pos,optixLaunchParams.camera.dir_du),dot(center - optixLaunchParams.camera.pos,optixLaunchParams.camera.dir_dv));
    //
    // int centerx = int(ceneter_screen_space.x * self.frameBufferSize.x);
    // int centery = int(ceneter_screen_space.y * self.frameBufferSize.y);
    vec3f dir_00 = optixLaunchParams.camera.dir_00 + 0.5f * (optixLaunchParams.camera.dir_du + optixLaunchParams.camera.dir_dv);
    vec2i center_screen_space = ComputeScreenSpace(optixLaunchParams.camera.pos,dir_00,optixLaunchParams.camera.dir_du,optixLaunchParams.camera.dir_dv,center,self.frameBufferSize);
    int centerx = center_screen_space.x;
    int centery = center_screen_space.y;

   // printf("center %d %d %d %d\n",centerx,centery,pixelId.x,pixelId.y);

    int pi = pixelId.x;
    int pj = pixelId.y;
    float radius = Distance(pi,pj,centerx,centery);
  // printf("center %d %d %d %d %f\n",pi,pj,centerx,centery,radius);
    // radius = max(radius,5.f);
    float theta = calculateTheta(vec2f(centerx,centery),vec2f(pi,pj));

    float maxTheta = optixLaunchParams.thetaRange;
    float deltaTheta = maxTheta / 10.f;

    float curTheta = -maxTheta;

    float totalWeight = 0.0f;

    vec3f centerColor = vec3f(optixLaunchParams.denoiseBuffer[fbOfs]);
    
    vec3f color = vec3f(0.f,0.f,0.f);
    int filterCount = 0;
    float filterWeight[100];

    int lastCurX  = 100000;
    int lastCurY = 100000;
    
    while(curTheta < maxTheta) {
        
        float curX = centerx + radius * cos(theta + curTheta);
        float curY = centery + radius * sin(theta + curTheta);
        int curI = int(curX);
        int curJ = int(curY);

        if(curI == lastCurX && curJ == lastCurY) {
            curTheta += deltaTheta;
            lastCurX = curI;
            lastCurY = curJ;
            continue;
        }
        if(curI < 0 || curI >= self.frameBufferSize.x || curJ < 0 || curJ >= self.frameBufferSize.y) {
            curTheta += deltaTheta;
            continue;
        }
        int curOfs = curI + self.frameBufferSize.x * curJ;

        if(optixLaunchParams.gBuffer[curOfs].hit == false || optixLaunchParams.fittedGuffer[curOfs].strand_index==-100) {
            curTheta += deltaTheta;
            continue;
        }
        vec3f curpixelColor = vec3f(optixLaunchParams.denoiseBuffer[curOfs]);
        float weight = GetFilterWeight(optixLaunchParams.gBuffer[fbOfs],optixLaunchParams.gBuffer[curOfs],curpixelColor,centerColor,pixelId,vec2i(curI,curJ));

        if(!AlmostZero(weight)) {
            weight = max(optixLaunchParams.weightThreshold,weight);
            // printf("Weight %f\n",weight);
        }
        
        totalWeight += weight;
        color += weight * curpixelColor;
        curTheta += deltaTheta;
        lastCurX = curI;
        lastCurY = curJ;
    }
    if(!AlmostZero(totalWeight))
        color /= totalWeight;
    else
        color = centerColor;

    if(optixLaunchParams.debugMode == VisualizeNormal)
        color = 0.5f * optixLaunchParams.gBuffer[fbOfs].normal + 0.5f;
    if(optixLaunchParams.debugMode == VisualizeTangent)
        color = 0.5f * optixLaunchParams.gBuffer[fbOfs].tangent + 0.5f;
    if(optixLaunchParams.debugMode == VisualizePosition)
        color =  optixLaunchParams.gBuffer[fbOfs].position;
    if(optixLaunchParams.debugMode == VisualizeBlack)
        color = vec3f(0.f,0.f,0.f);

    //printf("weight %f %f %f %f %f %f %f %f %f %f\n",filterWeight[0],filterWeight[1],filterWeight[2],filterWeight[3],filterWeight[4],filterWeight[5],filterWeight[6],filterWeight[7],filterWeight[8],filterWeight[9]);
    //color = optixLaunchParams.gBuffer[fbOfs].normal;
    // printf("Filter count %d\n",filterCount);
    // optixLaunchParams.denoiseBuffer[fbOfs] = make_float4(color.x,color.y,color.z,1.0f);
    // writePixel(optixLaunchParams.denoiseBuffer,self.frameBuffer,fbOfs);

    optixLaunchParams.accumBuffer[fbOfs].x -= centerColor.x;
    optixLaunchParams.accumBuffer[fbOfs].y -= centerColor.y;
    optixLaunchParams.accumBuffer[fbOfs].z -= centerColor.z;
    
    writePixel(color, optixLaunchParams.accumId,
       self.frameBuffer,
       optixLaunchParams.accumBuffer,
       optixLaunchParams.averageBuffer,
       fbOfs);
}