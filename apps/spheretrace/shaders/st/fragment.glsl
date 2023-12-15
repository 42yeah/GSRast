#version 430 core

in vec3 boxNormal;
in vec3 worldPos;

out vec4 color;

uniform vec3 camPos;
uniform vec3 sphereCenter;
uniform vec3 sphereScale;
uniform mat3 sphereRotation;
uniform bool cubeMode;

#define HI_PRECISION

#ifdef HI_PRECISION
#define VEC dvec3
#define FLT double
#define MAT dmat3
#else
#define VEC vec3
#define FLT float
#define MAT mat3
#endif

/**
 * This function checks for whether (p - ro) intersects a sphere
 * located at c, with radius r = 1.0.
 */
vec3 sphereIntersect(vec3 c, vec3 ro, vec3 p, out vec3 normal) {
    MAT sphereRotationT = transpose(sphereRotation);

    VEC rd = VEC(sphereRotationT * normalize(p - ro)) / VEC(sphereScale);
    VEC oRel = (sphereRotationT * VEC(ro - c)) / VEC(sphereScale); // ro relative to c

    FLT a = dot(rd, rd);
    FLT b = 2.0 * dot(oRel, rd);
    FLT cc = dot(oRel, oRel) - 1.0;

    FLT discriminant = b * b - 4 * a * cc;

    // no intersection
    if (discriminant < 0.0) {
        return vec3(0.0);
    }

    // yes intersection - but do we need to calculate t? or...
    FLT t1 = (-b + sqrt(discriminant)) / (2.0 * a);
    FLT t2 = (-b - sqrt(discriminant)) / (2.0 * a);
    FLT t = min(t1, t2);
    vec3 intersection = ro + sphereRotation * (vec3(t * rd) * sphereScale);

    normal = normalize((intersection - c) / sphereScale);
    return intersection;
}

void main() {
    vec3 nor = vec3(0.0);
    vec3 sp = sphereIntersect(sphereCenter, camPos, worldPos, nor);
    if (!cubeMode && sp == vec3(0.0)) {
        discard;
    } else if (cubeMode && sp == vec3(0.0)) {
        float boxCol = max(dot(normalize(vec3(1.0, 2.0, 3.0)), boxNormal), 0.0) * 0.8 + 0.2;
        color = vec4(boxCol * vec3(1.0, 0.0, 1.0), 1.0);
        return;
    }

    float col = max(dot(normalize(vec3(1.0, 2.0, 3.0)), nor), 0.0);
    color = vec4(col * vec3(1.0, 0.5, 0.0), 1.0);
}
