namespace SphereTriangleDivisionHP;

internal static class LatticeGrid {
    public static LatticeKey[] BuildPointKeys(int n) {
        List<LatticeKey> keys = [];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n - i; j++) {
                keys.Add(new LatticeKey(i, j));
            }
        }

        return keys.ToArray();
    }

    public static TriangleKey[] BuildTriangleKeys(int n) {
        List<TriangleKey> triangles = [];

        for (int i = 0; i <= n - 1; i++) {
            for (int j = 0; j <= n - 1 - i; j++) {
                int k = n - i - j;
                LatticeKey a = new(i, j);
                LatticeKey b = new(i + 1, j);
                LatticeKey c = new(i, j + 1);
                triangles.Add(new TriangleKey(a, b, c));

                if (k >= 2) {
                    LatticeKey d = new(i + 1, j + 1);
                    triangles.Add(new TriangleKey(b, d, c));
                }
            }
        }

        return triangles.ToArray();
    }

    public static (ConstraintMode mode, int axis) ClassifyVertexConstraint(LatticeKey key, int n) {
        int k = n - key.I - key.J;
        int zeroCount = (key.I == 0 ? 1 : 0) + (key.J == 0 ? 1 : 0) + (k == 0 ? 1 : 0);

        if (zeroCount >= 2) {
            return (ConstraintMode.Corner, -1);
        }
        if (key.I == 0) {
            return (ConstraintMode.Edge, 0);
        }
        if (key.J == 0) {
            return (ConstraintMode.Edge, 1);
        }
        if (k == 0) {
            return (ConstraintMode.Edge, 2);
        }

        return (ConstraintMode.Interior, -1);
    }
}
