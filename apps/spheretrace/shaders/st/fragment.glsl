#version 430 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

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
    vec3 localIntersection = ((mat3(sphereRotationT) * (intersection - c)) / sphereScale);

    normal = sphereRotation * localIntersection;
    return intersection;
}

void main() {
    vec3 nor = vec3(0.0);
    vec3 sp = sphereIntersect(sphereCenter, camPos, worldPos, nor);

    color = vec4(1.0, 1.0, 1.0, 1.0);
    if (!cubeMode && sp == vec3(0.0)) {
        discard;
    } else if (cubeMode && sp == vec3(0.0)) {
        float boxCol = max(dot(normalize(vec3(0.31, 0.87, 0.1)), boxNormal), 0.0) * 0.4 + 0.6;
        color = vec4(boxCol * vec3(0.424, 0.557, 0.749), 1.0);
        // return;
    }

    mat4 scl = mat4(sphereScale.x, 0.0, 0.0, 0.0,
                    0.0, sphereScale.y, 0.0, 0.0,
                    0.0, 0.0, sphereScale.z, 0.0,
                    0.0, 0.0, 0.0, 1.0);
    mat4 onlyFour = mat4(1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         camPos.x, camPos.y, camPos.z, 1.0);
    // Axis-aligned projection
    mat4 aaProj = perspective * onlyFour * scl * mat4(1.0, 0.0, 0.0, 1.0,
                                           0.0, 1.0, 0.0, 1.0,
                                           0.0, 0.0, 1.0, 1.0,
                                           0.0, 0.0, 0.0, 1.0);
    vec4 pa = aaProj[0];
    vec4 pb = aaProj[1];
    vec4 pc = aaProj[2];
    pa /= pa.w;
    pb /= pb.w;
    pc /= pc.w;
    vec2 paa = vec2(pa);
    vec2 pbb = vec2(pb);
    vec2 pcc = vec2(pc);
    float aLength = length(paa);
    float bLength = length(pbb);
    float cLength = length(pcc);

    mat4 proj = perspective * view * model * mat4(1.0, 0.0, 0.0, 1.0,
                                                  0.0, 1.0, 0.0, 1.0,
                                                  0.0, 0.0, 1.0, 1.0,
                                                  0.0, 0.0, 0.0, 1.0);
    vec4 a = proj[0];
    vec4 b = proj[1];
    vec4 c = proj[2];
    a /= a.w;
    b /= b.w;
    c /= c.w;

    vec4 projPoint = perspective * view * model * vec4(sphereCenter, 1.0);
    projPoint /= projPoint.w;

    vec2 uv = vec2(gl_FragCoord) / vec2(800, 800);
    uv = uv * 2.0 - 1.0;
    // color = vec4(uv, 0.0, 1.0);

    vec2 aa = vec2(a);
    vec2 bb = vec2(b);
    vec2 cc = vec2(c);
    if (length(aa) < length(cc) && abs(dot(normalize(aa), normalize(cc))) - 1.0 >= 0.01) {
        vec2 swp = aa;
        aa = cc;
        cc = swp;

        float swapL = aLength;
        aLength = cLength;
        cLength = swapL;

    }
    if (length(bb) < length(cc) && abs(dot(normalize(bb), normalize(cc))) - 1.0 >= 0.01) {
        bb = cc;
        bLength = cLength;
    }
    aa = normalize(aa) * aLength;
    bb = normalize(bb) * bLength;
    float prx = dot(normalize(aa), uv) / length(aa);
    float pry = dot(normalize(bb), uv) / length(bb);

    if (prx * prx + pry * pry > 1.0) {
        color *= vec4(0.5, 0.5, 0.5, 1.0);
    } else {
        color *= vec4(0.9, 0.9, 0.9, 1.0);
    }
    if (abs(dot(normalize(aa), normalize(uv))) > 0.99) {
        if (abs(dot(normalize(aa), uv)) < length(aa)) {
            color *= vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
    if (abs(dot(normalize(bb), normalize(uv))) > 0.99) {
        if (abs(dot(normalize(bb), uv)) < length(bb)) {
            color *= vec4(0.0, 1.0, 0.0, 1.0);
        }
    }
    if (distance(vec2(projPoint), uv) < 0.1) {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    }
    return;


    float col = clamp(0.6 + 0.4 * max(dot(normalize(vec3(0.31, 0.87, 0.1)), nor), 0.0), 0.0, 1.0);
    color = vec4(col * vec3(0.9, 0.9, 0.9), 1.0);
}
