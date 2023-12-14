/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#version 430

in mat4 MVP;
uniform vec3 camPos;

in vec3 worldPos;
in vec3 ellipsoidCenter;
in vec3 ellipsoidScale;
in mat3 ellipsoidRotation;
in vec3 colorVert;
in flat int boxID;

out vec4 out_color;

vec3 closestEllipsoidIntersection(vec3 rayDirection, out vec3 normal) {
  // Convert ray to ellipsoid space
  dvec3 localRayOrigin = (camPos - ellipsoidCenter) * ellipsoidRotation;
  dvec3 localRayDirection = normalize(rayDirection * ellipsoidRotation);

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
  normal = normalize(ellipsoidRotation * localNormal);

  // Convert intersection point back to world space
  vec3 intersection = ellipsoidRotation * localIntersection + ellipsoidCenter;

  return intersection;
}

void main(void) {
    if (worldPos == vec3(0.0)) {
        discard;
    }

	vec3 dir = normalize(worldPos - camPos);

	vec3 normal;
	vec3 intersection = closestEllipsoidIntersection(dir, normal);
	float align = max(0.4, dot(-dir, normal));

	out_color = vec4(1, 0, 0, 1);

	if(intersection == vec3(0))
		discard;

	vec4 newPos = MVP * vec4(intersection, 1);
	newPos /= newPos.w;

	gl_FragDepth = newPos.z;

	float a = 1.0;

	out_color = vec4(align * colorVert, a);
}
