#version 430 core

in vec3 worldPos;

out vec4 color;

uniform vec3 camPos;
uniform vec3 sphereCenter;

#define HI_PRECISION

#ifdef HI_PRECISION
#define VEC dvec3
#define FLT double
#else
#define VEC vec3
#define FLT float
#endif

/**
 * This function checks for whether (p - ro) intersects a sphere
 * located at c, with radius r = 1.0.
 */
vec3 sphereIntersect(vec3 c, vec3 ro, vec3 p, out vec3 normal) {
    VEC rd = VEC(normalize(p - ro));
    VEC oRel = VEC(ro - c); // ro relative to c

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
    vec3 intersection = ro + vec3(t * rd);

    normal = normalize(intersection - c);
    return intersection;
}

void main() {
    vec3 nor = vec3(0.0);
    vec3 sp = sphereIntersect(sphereCenter, camPos, worldPos, nor);
    if (sp == vec3(0.0)) {
        discard;
    }

    float col = max(dot(vec3(1.0, 1.0, 1.0), nor), 0.0);
    color = vec4(col * vec3(1.0, 0.5, 0.0), 1.0);
}
