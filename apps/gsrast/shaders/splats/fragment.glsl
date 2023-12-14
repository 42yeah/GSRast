#version 430 core

in vec3 position;
in vec3 ellipsoidCenter;
in vec3 ellipsoidScale;
in mat3 ellipsoidRot;
in vec3 color;

in mat4 vModel;
in mat4 vView;
in mat4 vPerspective;

out vec4 outColor;

uniform vec3 camPos;
uniform vec3 camFront;


// Ray-sphere intersection
// c: Ellipsoid center,
// d: Ray direction (position - camera eye)
vec3 rsIntersection(out vec3 normal) {
    vec3 localRo = (camPos - ellipsoidCenter) * ellipsoidRot;
    vec3 localRd = normalize(normalize(position - camPos) * ellipsoidRot);

    vec3 oneOver = 1.0 / ellipsoidScale;

    vec3 localRoScl = localRo * oneOver;
    vec3 localRdScl = localRd * oneOver;

    float a = dot(localRdScl, localRdScl);
    float b = 2.0 * dot(localRoScl, localRdScl);
    float c = dot(localRoScl, localRoScl) - 1.0; // We have normalized the radius by dividing it by scale

    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return vec3(0.0);
    }

    float t1 = (-b + sqrt(discriminant)) / (2.0 * a);
    float t2 = (-b - sqrt(discriminant)) / (2.0 * a);
    float t = min(t1, t2);

    vec3 localPos = localRo + t * localRd;
    vec3 localN = normalize(localPos / ellipsoidScale);
    normal = normalize(ellipsoidRot * localN);

    vec3 intersection = ellipsoidRot * localPos + ellipsoidCenter;
    return intersection;
}

void main() {
    vec3 normal = vec3(0.0);
    vec3 intersection = rsIntersection(normal);
    if (intersection == vec3(0.0)) {
        discard;
    }

    vec4 newPos = vPerspective * vView * vModel * vec4(intersection, 1.0);
    newPos /= newPos.w;
    gl_FragDepth = newPos.z;

    outColor = vec4(color, 1.0);
}
