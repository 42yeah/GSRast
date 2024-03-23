#version 430

layout (location = 0) in vec3 aPos;

layout (std430, binding = 0) buffer splatPosition {
    vec4 positions[];
};
layout (std430, binding = 1) buffer splatScale {
    vec4 scales[];
};
layout (std430, binding = 2) buffer splatColor {
    vec4 colors[];
};
layout (std430, binding = 3) buffer splatQuat {
    vec4 quats[];
};
layout (std430, binding = 4) buffer splatAlpha {
    float alphas[];
};

uniform mat4 view;
uniform mat4 perspective;

out vec3 position;
out vec3 ellipsoidCenter;
out vec3 ellipsoidScale;
out mat3 ellipsoidRot;
out float ellipsoidAlpha;
out vec4 conicOpacity;

out vec3 color;


void computeCov3D(vec3 scale, float scaleModifier, const vec4 rotation, out float cov3Ds[6])
{
    // 1. Make scaling matrix
    mat3 scalingMat = mat3(1.0f);
    for (int i = 0; i < 3; i++)
    {
	scalingMat[i][i] = scaleModifier * scale[i]; // I love glm
    }

    // 2. Transfer quaternion to rotation matrix
    mat3 rotMat = quatToMat(normalize(rotation)); // They are normalized when they come in, so.

    // 3. According to paper, cov = R S S R
    mat3 rs = rotMat * scalingMat;
    mat3 sigma = rs * transpose(rs);

    // 4. Sigma is symmetric, only store upper right
    // _ _ _
    // x _ _
    // x x _
    // I think there is an error here in the original SIBR code, but it doesn't matter anyway since its symmetric
    cov3Ds[0] = sigma[0][0];
    cov3Ds[1] = sigma[1][0];
    cov3Ds[2] = sigma[2][0];
    cov3Ds[3] = sigma[1][1];
    cov3Ds[4] = sigma[2][1];
    cov3Ds[5] = sigma[2][2];
}

vec3 computeCov2D(vec3 mean, float focalDist, float tanFOVx, float tanFOVy, float cov3D[6], mat4 viewMat)
{
    // Follows Eq. 29 and Eq. 31 of EWA splatting, with focal length taken into consideration.
    // 1. Transform point center to camera space (t)
    vec3 t = vec3(viewMat * vec4(mean, 1.0f));

    float limx = 1.3f * tanFOVx;
    float limy = 1.3f * tanFOVy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;

    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    // 2. Calculate Jacobian.
    mat3 jacobian = mat3(focalDist / t.z, 0.0f, (-focalDist * t.x) / (t.z * t.z),
				   0.0f, focalDist / t.z, (-focalDist * t.y) / (t.z * t.z),
				   0.0f, 0.0f, 0.0f);
    mat3 view3x3 = transpose(viewMat);
    mat3 T = view3x3 * jacobian;

    // Recover the covariance matrix.
    mat3 Vrk = mat3(cov3D[0], cov3D[1], cov3D[2],
			      cov3D[1], cov3D[3], cov3D[4],
			      cov3D[2], cov3D[4], cov3D[5]);

    // J * W * Vrk * W^T * J^T
    mat3 cov = transpose(T) * Vrk * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    // ???
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

mat3 quatToMat(vec4 q) {
    return mat3(2.0 * (q.x * q.x + q.y * q.y) - 1.0, 2.0 * (q.y * q.z + q.x * q.w), 2.0 * (q.y * q.w - q.x * q.z), // 1st column
		2.0 * (q.y * q.z - q.x * q.w), 2.0 * (q.x * q.x + q.z * q.z) - 1.0, 2.0 * (q.z * q.w + q.x * q.y), // 2nd column
		2.0 * (q.y * q.w + q.x * q.z), 2.0 * (q.z * q.w - q.x * q.y), 2.0 * (q.x * q.x + q.w * q.w) - 1.0); // last column
}

void main() {
    vec3 scale = vec3(scales[gl_InstanceID]) * 2.0;
    vec3 scaled = scale * aPos;
    mat3 rot = quatToMat(quats[gl_InstanceID]);

    vec3 rotated = rot * scaled;
    vec3 posOffset = rotated + vec3(positions[gl_InstanceID]);
    vec4 mPos = vec4(posOffset, 1.0);

    position = vec3(mPos);
    ellipsoidCenter = vec3(positions[gl_InstanceID]);
    ellipsoidScale = scale;
    ellipsoidRot = rot;
    ellipsoidAlpha = alphas[gl_InstanceID];

    float cov3Ds[6];
    computeCov3D(scale, 1.0, rot, cov3Ds);
    vec3 cov = computeCov2D(positions[gl_InstanceID], 1.0, 1.0, 1.0, cov3Ds, view);

    gl_Position = perspective * view * mPos;
    color = vec3(colors[gl_InstanceID]) * 0.2 + vec3(0.5);
}
