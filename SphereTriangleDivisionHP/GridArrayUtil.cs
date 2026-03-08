using DoubleDoubleGeometry.Geometry3D;

namespace SphereTriangleDivisionHP;

internal static class GridArrayUtil {
    public static Vector3D[,] CreateZeroFilled(int ni, int nj) {
        Vector3D[,] grid = new Vector3D[ni, nj];

        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nj; j++) {
                grid[i, j] = Vector3D.Zero;
            }
        }

        return grid;
    }

    public static Vector3D[,] ClonePositions(Vector3D[,] positions) {
        int ni = positions.GetLength(0);
        int nj = positions.GetLength(1);
        Vector3D[,] clone = CreateZeroFilled(ni, nj);

        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nj; j++) {
                clone[i, j] = positions[i, j];
            }
        }

        return clone;
    }

    public static Vector3D RequirePoint(Vector3D[,] positions, LatticeKey key) => positions[key.I, key.J];
}
