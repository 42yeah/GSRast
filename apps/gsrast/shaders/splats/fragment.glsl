#version 430 core

in vec3 position;
in vec3 ellipsoidCenter;
in vec3 ellipsoidScale;
in float ellipsoidAlpha;
in mat3 ellipsoidRot;
in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec4 outColor;

uniform vec3 camPos;
uniform vec3 camFront;

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
    MAT ellipoidRotT = transpose(ellipsoidRot);

    VEC rd = VEC(ellipoidRotT * normalize(p - ro)) / VEC(ellipsoidScale);
    VEC oRel = (ellipoidRotT * VEC(ro - c)) / VEC(ellipsoidScale); // ro relative to c

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
    vec3 intersection = ro + ellipsoidRot * (vec3(t * rd) * ellipsoidScale);
    vec3 localIntersection = ((mat3(ellipoidRotT) * (intersection - c)) / ellipsoidScale);

    normal = ellipsoidRot * localIntersection;
    return intersection;
}


void main() {
    if (ellipsoidAlpha < 0.3) {
        discard;
    }

    vec3 normal = vec3(0.0);
    vec3 intersection = sphereIntersect(ellipsoidCenter, camPos, position, normal);
    if (intersection == vec3(0.0)) {
        discard;
    }

    vec3 rd = normalize(camPos - intersection);
    float align = max(dot(rd, normal), 0.1);

    vec4 newPos = perspective * view * model * vec4(intersection, 1.0);
    newPos /= newPos.w;
    gl_FragDepth = newPos.z;

    outColor = vec4(align * color, 1.0);
}
