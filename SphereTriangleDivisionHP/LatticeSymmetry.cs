using DoubleDouble;
using DoubleDoubleGeometry.Geometry3D;

namespace SphereTriangleDivisionHP;

internal static class LatticeSymmetry {
    public static readonly int[][] Permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    public static ((int, int, int), Vector3D) CanonicalizeTriplet(int i, int j, int k, Vector3D xyz) {
        int[] idx = [i, j, k];
        List<((int, int, int) key, Vector3D xyz)> candidates = [];

        foreach (int[] p in Permutations) {
            int i0 = idx[p[0]];
            int i1 = idx[p[1]];
            int i2 = idx[p[2]];

            if (i0 > i1) {
                continue;
            }
            if (i1 == i2 && i0 != i1) {
                continue;
            }

            candidates.Add(((i0, i1, i2), Permute(xyz, p)));
        }

        if (candidates.Count == 0) {
            return ((i, j, k), xyz);
        }

        return candidates
            .OrderBy(v => v.key.Item1 == v.key.Item2 ? 0 : 1)
            .ThenBy(v => v.key.Item1)
            .ThenBy(v => v.key.Item2)
            .ThenBy(v => v.key.Item3)
            .First();
    }

    public static Vector3D Permute(Vector3D v, int[] p) {
        ddouble[] values = [v.X, v.Y, v.Z];
        return new Vector3D(values[p[0]], values[p[1]], values[p[2]]);
    }
}
