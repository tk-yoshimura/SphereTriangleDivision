using DoubleDouble;
using DoubleDoubleGeometry.Geometry3D;

namespace SphereTriangleDivisionHP;

internal static class SphericalGeometry {
    public static readonly ddouble AreaEps = ddouble.Parse("1e-60");

    public static ddouble[] ComputeTriangleAreas(Vector3D[,] positions, TriangleKey[] triangleKeys) {
        ddouble[] areas = new ddouble[triangleKeys.Length];

        for (int i = 0; i < triangleKeys.Length; i++) {
            TriangleKey tri = triangleKeys[i];
            areas[i] = SphericalTriangleArea(
                GridArrayUtil.RequirePoint(positions, tri.A),
                GridArrayUtil.RequirePoint(positions, tri.B),
                GridArrayUtil.RequirePoint(positions, tri.C)
            );
        }

        return areas;
    }

    public static ddouble SphericalTriangleArea(Vector3D a, Vector3D b, Vector3D c) {
        a = Normalize(a);
        b = Normalize(b);
        c = Normalize(c);

        ddouble det = ddouble.Abs(Vector3D.Dot(a, Vector3D.Cross(b, c)));
        ddouble denom = 1d + Vector3D.Dot(a, b) + Vector3D.Dot(b, c) + Vector3D.Dot(c, a);

        return 2d * ddouble.Atan2(det, ddouble.Max(denom, AreaEps));
    }

    public static Vector3D Normalize(Vector3D v) {
        if (v.Norm <= 0d) {
            throw new InvalidOperationException("zero vector cannot be normalized");
        }

        return v.Normal;
    }

    public static Vector3D Max(Vector3D v, ddouble lower) =>
        new(ddouble.Max(v.X, lower), ddouble.Max(v.Y, lower), ddouble.Max(v.Z, lower));
}
