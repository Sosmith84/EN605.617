// polynomial_transform.cl

__kernel void poly_transform(__global const float* in,
                             __global float4* out,
                             const int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;

    float x = in[gid];
    float x2 = x * x;
    float x3 = x2 * x;
    float s  = sin(x);

    // Store (x, x^2, x^3, sin(x)) in a float4
    out[gid] = (float4)(x, x2, x3, s);
}
