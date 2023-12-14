#version 430 core

in vec3 position;
in vec3 ellipsoidCenter;
in vec3 ellipsoidScale;
in mat3 ellipsoidRot;
in vec3 color;

in mat4 MVP;

out vec4 outColor;

uniform vec3 camPos;
uniform vec3 camFront;


// Ray-sphere intersection
// c: Ellipsoid center,
// d: Ray direction (position - camera eye)
// vec3 rsIntersection(out vec3 normal) {
//     vec3 localRo = (camPos - ellipsoidCenter) * ellipsoidRot;
//     vec3 localRd = normalize(normalize(position - camPos) * ellipsoidRot);
//
//     vec3 oneOver = 1.0 / ellipsoidScale;
//
//     vec3 localRoScl = localRo * oneOver;
//     vec3 localRdScl = localRd * oneOver;
//
//     float a = dot(localRdScl, localRdScl);
//     float b = 2.0 * dot(localRoScl, localRdScl);
//     float c = dot(localRoScl, localRoScl) - 1.0; // We have normalized the radius by dividing it by scale
//
//     float discriminant = b * b - 4.0 * a * c;
//
//     if (discriminant < 0.0) {
//         return vec3(0.0);
//     }
//
//     float t1 = (-b + sqrt(discriminant)) / (2.0 * a);
//     float t2 = (-b - sqrt(discriminant)) / (2.0 * a);
//     float t = min(t1, t2);
//
//     vec3 localPos = localRo + t * localRd;
//     vec3 localN = normalize(localPos / ellipsoidScale);
//     normal = normalize(ellipsoidRot * localN);
//
//     vec3 intersection = ellipsoidRot * localPos + ellipsoidCenter;
//     return intersection;
// }

vec3 closestEllipsoidIntersection(vec3 rayDirection, out vec3 normal) {
  // Convert ray to ellipsoid space
  dvec3 localRayOrigin = (camPos - ellipsoidCenter) * ellipsoidRot;
  dvec3 localRayDirection = normalize(rayDirection * ellipsoidRot);

  dvec3 oneover = double(1) / dvec3(ellipsoidScale);

  // Compute coefficients of quadratic equation
  double a = dot(localRayDirection * oneover, localRayDirection * oneover);
  double b = 2.0 * dot(localRayDirection * oneover, localRayOrigin * oneover);
  double c = dot(localRayOrigin * oneover, localRayOrigin * oneover) - 1.0;

  // Compute discriminant
  double discriminant = b * b - 4.0 * a * c;

  // If discriminant is negative, there is no intersection
  if (discriminant < 0.0) {
    return vec3(0.0);
  }

  // Compute two possible solutions for t
  float t1 = float((-b - sqrt(discriminant)) / (2.0 * a));
  float t2 = float((-b + sqrt(discriminant)) / (2.0 * a));

  // Take the smaller positive solution as the closest intersection
  float t = min(t1, t2);

  // Compute intersection point in ellipsoid space
  vec3 localIntersection = vec3(localRayOrigin + t * localRayDirection);

  // Compute normal vector in ellipsoid space
  vec3 localNormal = normalize(localIntersection / ellipsoidScale);

  // Convert normal vector to world space
  normal = normalize(ellipsoidRot * localNormal);

  // Convert intersection point back to world space
  vec3 intersection = ellipsoidRot * localIntersection + ellipsoidCenter;

  return intersection;
}

void main() {
    vec3 localRd = normalize(position - camPos);
    vec3 normal = vec3(0.0);
    vec3 intersection = closestEllipsoidIntersection(localRd, normal);
    if (intersection == vec3(0.0)) {
        discard;
    }

    vec4 newPos = MVP * vec4(intersection, 1.0);
    newPos /= newPos.w;
    gl_FragDepth = newPos.z;

    outColor = vec4(color, 1.0);
}
